import tensorflow as tf
import numpy as np
#import keras
#from keras import backend as K
#from keras.models import Sequential
#from keras.layers.core import Layer, Dense, Lambda
import getdist
from getdist import plots, MCSamples
import emcee
import matplotlib.pyplot as plt
from tqdm import tnrange

class DelfiMixtureDensityNetwork():

    def __init__(self, simulator, prior, asymptotic_posterior, \
                 Finv, theta_fiducial, data, n_components, \
                 simulator_args = None, n_hidden = [50, 50], \
                 activations = [tf.nn.tanh, tf.nn.tanh], η = 0.01, names=None, \
                 labels=None, ranges=None, nwalkers=100, \
                 posterior_chain_length=1000, proposal_chain_length=100, \
                 rank=0, n_procs=1, comm=None, red_op=None, \
                 show_plot=True):

        # Input x and output t dimensions
        self.D = len(data)
        self.npar = len(theta_fiducial)

        # Number of GMM components
        self.M = n_components

        # Total number of outputs for the neural network
        self.N = int((self.D + self.D * (self.D + 1) / 2 + 1)*self.M)

        # Number of hiden units and activations
        self.n_hidden = n_hidden
        self.activations = activations
        self.η = η

        # Build TensorFlow model
        self.build_network()

        # Prior and asymptotic posterior
        self.prior = prior
        self.asymptotic_posterior = asymptotic_posterior

        # Training data
        self.ps = []
        self.xs = []
        self.n_sims = 0

        # MCMC chain parameters
        self.nwalkers = nwalkers
        self.posterior_chain_length = posterior_chain_length
        self.proposal_chain_length = proposal_chain_length

        # MCMC samples of learned posterior
        self.posterior_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.posterior_chain_length)])
        self.proposal_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.proposal_chain_length)])

        # Simulator
        self.simulator = simulator
        self.simulator_args = simulator_args

        # Fisher matrix and fiducial parameters
        self.Finv = Finv
        self.fisher_errors = np.sqrt(np.diag(self.Finv))
        self.theta_fiducial = theta_fiducial

        # Data
        self.data = data

        # Parameter names and ranges for plotting with GetDist
        self.names = names
        self.labels = labels
        self.ranges = ranges
        self.show_plot = show_plot

        # Training loss, validation loss
        self.loss = []
        self.val_loss = []
        self.loss_trace = []
        self.val_loss_trace = []
        self.n_sim_trace = []

        # MPI-specific setup
        self.rank = rank
        self.n_procs = n_procs
        if n_procs > 1:
            self.use_mpi = True
            self.comm = comm
            self.red_op = red_op
        else:
            self.use_mpi = False

    # Divide list of jobs between MPI processes
    def allocate_jobs(self, n_jobs):

        n_j_allocated = 0
        for i in range(self.n_procs):
            n_j_remain = n_jobs - n_j_allocated
            n_p_remain = self.n_procs - i
            n_j_to_allocate = n_j_remain // n_p_remain
            if self.rank == i:
                return range(n_j_allocated, \
                             n_j_allocated + n_j_to_allocate)
            n_j_allocated += n_j_to_allocate

    # Combine arrays from all processes assuming
    # 1) array was initially zero
    # 2) each process has edited a unique slice of the array
    def complete_array(self, target_distrib):

        if self.use_mpi:
            target = np.zeros(target_distrib.shape, \
                              dtype=target_distrib.dtype)
            self.comm.Allreduce(target_distrib, target, \
                                op=self.red_op)
        else:
            target = target_distrib
        return target

    # Log posterior
    def log_posterior(self, x):

        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return self.log_likelihood((x - self.theta_fiducial)/self.fisher_errors) + np.log(self.prior.pdf(x))

    # Log posterior
    def log_geometric_mean_proposal(self, x):

        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return 0.5 * (self.log_likelihood((x - self.theta_fiducial)/self.fisher_errors) + 2 * np.log(self.prior.pdf(x)) )

    # Run n_batch simulations
    def run_simulation_batch(self, n_batch, ps):

        # Dimension outputs
        data_samples = np.zeros((n_batch, self.npar))
        parameter_samples = np.zeros((n_batch, self.npar))

        # Run samples assigned to each process, catching exceptions
        # (when simulator returns np.nan).
        i_prop = self.inds_prop[0]
        i_acpt = self.inds_acpt[0]
        err_msg = 'Simulator returns {:s} for parameter values: {} (rank {:d})'
        while i_acpt <= self.inds_acpt[-1]:
            try:
                sim = self.simulator(ps[i_prop,:], self.simulator_args)
                if np.all(np.isfinite(sim.flatten())):
                    data_samples[i_acpt,:] = sim
                    parameter_samples[i_acpt,:] = ps[i_prop,:]
                    i_acpt += 1
                else:
                    print(err_msg.format('NaN/inf', ps[i_prop,:], self.rank))
            except:
                print(err_msg.format('exception', ps[i_prop,:], self.rank))
            i_prop += 1

        # Reduce results from all processes and return
        data_samples = self.complete_array(data_samples)
        parameter_samples = self.complete_array(parameter_samples)
        return data_samples, parameter_samples

    # EMCEE sampler
    def emcee_sample(self, log_likelihood, x0, burn_in_chain=100, main_chain=100):

        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.D, log_likelihood)

        # Burn-in chain
        pos, prob, state = sampler.run_mcmc(x0, burn_in_chain)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(pos, main_chain)

        return sampler.flatchain

    # MDN log likelihood
    def log_likelihood(self, theta):

        like = self.sess.run(self.like, feed_dict = {self.parameter: theta.astype(np.float32).reshape((1, self.D)), self.true: ((self.data - self.theta_fiducial)/self.fisher_errors).astype(np.float32).reshape((1, self.D))})[0]
        if np.isnan(np.log(like)):
            return 1e-300
        else:
            return np.log(like)

    def sequential_training(self, n_initial, n_batch, n_populations, proposal, \
                            safety = 5, plot = True, batch_size=100, \
                            validation_split=0.1, epochs=100, epsilon = 1e-37):

        # Generate initial theta values from some broad proposal on
        # master process and share with other processes. Overpropose
        # by a factor of safety to (hopefully) cope gracefully with
        # the possibility of some bad proposals. Assign indices into
        # proposal array (self.inds_prop) and accepted arrays
        # (self.inds_acpt) to allow for easy MPI communication.
        if self.rank == 0:
            ps = np.array([proposal.draw() for i in range(safety * n_initial)])
        else:
            ps = np.zeros((safety * n_initial, self.npar))
        if self.use_mpi:
            self.comm.Bcast(ps, root=0)
        self.inds_prop = self.allocate_jobs(safety * n_initial)
        self.inds_acpt = self.allocate_jobs(n_initial)

        # Run simulations at those theta values
        if self.rank == 0:
            print('Running initial {} sims...'.format(n_initial))
        self.xs, self.ps = self.run_simulation_batch(n_initial, ps)
        if self.rank == 0:
            print('Done.')

        # Train on master only
        if self.rank == 0:

            # Construct the initial training-set
            self.ps = (self.ps - self.theta_fiducial)/self.fisher_errors
            self.xs = (self.xs - self.theta_fiducial)/self.fisher_errors
            self.n_sims = len(self.xs)

            ## Train the network on these initial simulations
            history = self.training(self.ps, self.xs, batch_size = batch_size, validation_split = validation_split, epochs = epochs, epsilon = epsilon)

            ## Update the loss and validation loss
            self.loss = history["loss"]
            self.val_loss = history["val_loss"]
            self.loss_trace.append(history["loss"][-1])
            self.val_loss_trace.append(history["val_loss"][-1])
            self.n_sim_trace.append(self.n_sims)

            # Generate posterior samples
            print('Sampling approximate posterior...')
            self.posterior_samples = \
                self.emcee_sample(self.log_posterior, \
                                  [self.posterior_samples[-i,:] for i in range(self.nwalkers)], \
                                  main_chain=self.posterior_chain_length)
            print('Done.')

            # If plot == True, plot the current posterior estimate
            if plot == True:
                self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='seq_train_post_0.pdf')

        # Loop through a number of populations
        for i in range(n_populations):

            # Propose theta values on master process and share with
            # other processes. Again, ensure we propose more sets of
            # parameters than needed to cope with bad params.
            if self.rank == 0:

                # Current population
                print('Population {}/{}'.format(i+1, n_populations))

                # Sample the current posterior approximation
                print('Sampling proposal density...')
                self.proposal_samples = \
                    self.emcee_sample(self.log_geometric_mean_proposal, \
                                      [self.proposal_samples[-j,:] for j in range(self.nwalkers)], \
                                      main_chain=self.proposal_chain_length)
                ps_batch = self.proposal_samples[-safety * n_batch:,:]
                print('Done.')

            else:
                ps_batch = np.zeros((safety * n_batch, self.npar))
            if self.use_mpi:
                self.comm.Bcast(ps_batch, root=0)

            # Run simulations
            self.inds_prop = self.allocate_jobs(safety * n_batch)
            self.inds_acpt = self.allocate_jobs(n_batch)
            if self.rank == 0:
                print('Running {} sims...'.format(n_batch))
            xs_batch, ps_batch = self.run_simulation_batch(n_batch, ps_batch)
            if self.rank == 0:
                print('Done.')

            # Train on master only
            if self.rank == 0:

                # Augment the training data
                ps_batch = (ps_batch - self.theta_fiducial)/self.fisher_errors
                xs_batch = (xs_batch - self.theta_fiducial)/self.fisher_errors
                self.ps = np.concatenate([self.ps, ps_batch])
                self.xs = np.concatenate([self.xs, xs_batch])
                self.n_sims += n_batch

                # Train the network on larger data set
                history = self.training(self.ps, self.xs, batch_size = batch_size, validation_split = validation_split, epochs = epochs, epsilon = epsilon)

                # Update the loss and validation loss
                self.loss = np.concatenate([self.loss, history['loss']])
                self.val_loss = np.concatenate([self.val_loss, history['val_loss']])
                self.loss_trace.append(history['loss'][-1])
                self.val_loss_trace.append(history['val_loss'][-1])
                self.n_sim_trace.append(self.n_sims)

                # Generate posterior samples
                print('Sampling approximate posterior...')
                self.posterior_samples = \
                    self.emcee_sample(self.log_posterior, \
                                      [self.posterior_samples[j] for j in range(self.nwalkers)], \
                                      main_chain=self.posterior_chain_length)
                print('Done.')

                # If plot == True
                if plot == True:
                    self.triangle_plot([self.posterior_samples], \
                                        savefig=True, \
                                        filename='seq_train_post_{:d}.pdf'.format(i + 1))

        # Train on master only
        if self.rank == 0:

            ## Train the network over some more epochs
            print('Final round of training with larger SGD batch size...')
            self.training(self.ps, self.xs, batch_size = self.n_sims, validation_split = 0.1, epochs = 300, epsilon = epsilon)
            print('Done.')

            print('Sampling approximate posterior...')
            self.posterior_samples = \
                self.emcee_sample(self.log_posterior, \
                                  [self.posterior_samples[-i,:] for i in range(self.nwalkers)], \
                                  main_chain=self.posterior_chain_length)
            print('Done.')

            # if plot == True
            if plot == True:
                self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='seq_train_post_final.pdf')

    def fisher_pretraining(self, n_batch, proposal, plot=True, batch_size=100, validation_split=0.1, epochs=100, epsilon = 1e-37):
        # Train on master only
        if self.rank == 0:

            # Generate fisher pre-training data
            print("Generating pre-training data...")

            # Anticipated covariance of the re-scaled data
            Cdd = np.zeros((self.npar, self.npar))
            for i in range(self.npar):
                for j in range(self.npar):
                    Cdd[i,j] = self.Finv[i,j]/(self.fisher_errors[i]*self.fisher_errors[j])
            Ldd = np.linalg.cholesky(Cdd)

            # Sample parameters from some broad proposal
            ps = np.zeros((n_batch, self.npar))
            for i in range(0, n_batch):
                ps[i,:] = (proposal.draw() - self.theta_fiducial)/self.fisher_errors

            # Sample data assuming a Gaussian likelihood
            xs = np.array([pss + np.dot(Ldd, np.random.normal(0, 1, self.npar)) for pss in ps])
            # Train network on initial (asymptotic) simulations
            print("Training on the pre-training data...")
            history = self.training(ps, xs, batch_size = batch_size, validation_split = validation_split, epochs = epochs, epsilon = epsilon)
            print("Done.")

            # Initialization for the EMCEE sampling
            x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]

            print('Sampling approximate posterior...')
            self.posterior_samples = self.emcee_sample(self.log_posterior, x0, main_chain=self.posterior_chain_length)
            print('Done.')

            # if plot == True
            if plot == True:
                self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='fish_pretrain_post.pdf')

            # Update the loss (as a function of the number of simulations) and number of simulations ran (zero so far)
            self.loss_trace.append(history['loss'][-1])
            self.val_loss_trace.append(history['val_loss'][-1])
            self.n_sim_trace.append(0)

    def triangle_plot(self, samples, savefig = False, filename = None):

        mc_samples = [MCSamples(samples=s, names = self.names, labels = self.labels, ranges = self.ranges) for s in samples]

        # Triangle plot
        g = plots.getSubplotPlotter(width_inch = 12)
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.6
        g.settings.axes_fontsize=14
        g.settings.legend_fontsize=16
        g.settings.lab_fontsize=20
        g.triangle_plot(mc_samples, filled_compare=True, normalized=True, legend_labels=['Density estimation likelihood-free inference'])
        for i in range(0, len(samples[0][0,:])):
            for j in range(0, i+1):
                ax = g.subplots[i,j]
                xtl = ax.get_xticklabels()
                ax.set_xticklabels(xtl, rotation=45)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)

        if savefig:
            print('Saving ' + filename)
            plt.savefig(filename)
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def training(self, input, true, batch_size=100, validation_split=0.1, epochs=100, epsilon = 1e-37):

        # make input and true data float32 for the build_network
        input = input.astype(np.float32)
        true = true.astype(np.float32)

        # Set up loss and val_loss container
        history = {"loss":[], "val_loss":[]}

        n_batch = input.shape[0]
        # Get indices of data for training on
        training_indices = np.arange(int(n_batch * (1 - validation_split)))
        training_batch_num = int(n_batch * (1 - validation_split)) // batch_size
        # Check whether data fits exactly into specified batch_size
        if training_batch_num * batch_size < len(training_indices):
            add_extra_training = True
        else:
            add_extra_training = False

        # Get indices of data for validation
        test_indices = np.arange(int(n_batch * (1 - validation_split)), n_batch)
        # Only validate if there is test data, if there is test data check
        # whether data fits exactly into specified batch_size
        if len(test_indices) == 0:
            validation = False
        else:
            validation = True
            test_batch_num = len(test_indices) // batch_size
            if test_batch_num * batch_size < len(test_indices):
                add_extra_test = True
            else:
                add_extra_test = False

        # Train for epochs number of epochs
        tr_epoch = tnrange(epochs, desc = "Epochs")
        for epoch in tr_epoch:

            # Shuffle the training indices at the start of each epoch so batches are different
            np.random.shuffle(training_indices)
            # Reshape input and true data into batches (all which fit into batch_size)
            these_indices = training_indices[:training_batch_num * batch_size]
            x_train = input[these_indices].reshape((training_batch_num, batch_size, self.npar))
            y_train = true[these_indices].reshape((training_batch_num, batch_size, self.npar))
            # Set up number of number of batches to train on
            if add_extra_training:
                num_training_batch = training_batch_num + 1
            else:
                num_training_batch = training_batch_num
            tr_batch = tnrange(num_training_batch, desc = "Batches")
            for batch in tr_batch:
                # Train on left over simulations (which didn't fit into batch_size)
                if batch == training_batch_num:
                    _, loss = self.sess.run([self.train, self.neg_log_normal_mixture_likelihood], feed_dict = {self.parameter: input[training_indices[training_batch_num * batch_size:]].reshape((len(training_indices[training_batch_num * batch_size:]), self.npar)), self.true: true[training_indices[training_batch_num * batch_size:]].reshape((len(training_indices[training_batch_num * batch_size:]), self.npar)), self.ϵ:epsilon})
                # Train on each batch
                else:
                    _, loss = self.sess.run([self.train, self.neg_log_normal_mixture_likelihood], feed_dict = {self.parameter: x_train[batch], self.true: y_train[batch], self.ϵ:epsilon})
                # After training, calculate loss on validation set
                if batch == num_training_batch - 1:
                    if validation:
                        val_loss = []
                        # Set up number of number of batches to validate on
                        if add_extra_test:
                            num_test_batch = test_batch_num + 1
                        else:
                            num_test_batch = test_batch_num
                        for test_batch in range(num_test_batch):
                            # Shuffle the test indices at the start of each epoch so batches are different
                            np.random.shuffle(test_indices)
                            these_test_indices = test_indices[:test_batch_num * batch_size]
                            x_test = input[these_test_indices].reshape((test_batch_num, batch_size, self.npar))
                            y_test = true[these_test_indices].reshape((test_batch_num, batch_size, self.npar))
                            if test_batch == test_batch_num:
                                # Calculate loss on validation set on left over simulations (which didn't fit into batch_size)
                                val_loss.append(self.sess.run(self.neg_log_normal_mixture_likelihood, feed_dict = {self.parameter: input[test_indices[test_batch_num * batch_size:]].reshape((len(test_indices[test_batch_num * batch_size:]), self.npar)), self.true: true[test_indices[test_batch_num * batch_size:]].reshape((len(test_indices[test_batch_num * batch_size:]), self.npar)), self.ϵ:epsilon}))
                            else:
                                # Calculate loss on validation set on each batch
                                val_loss.append(self.sess.run(self.neg_log_normal_mixture_likelihood, feed_dict = {self.parameter: x_test[test_batch], self.true: y_test[test_batch], self.ϵ:epsilon}))
                    else:
                        # If not validating, return nan
                        val_loss = np.nan
                    # Update progress bar
                    tr_batch.set_postfix(loss=loss, val_loss=np.mean(val_loss))
                else:
                    tr_batch.set_postfix(loss=loss)
            # Add loss of final batch and mean of all batches of validation loss to the history container at the end of each epoch
            history["loss"].append(loss)
            history["val_loss"].append(np.mean(val_loss))
        history["loss"] = np.array(history["loss"])
        history["val_loss"] = np.array(history["val_loss"])
        return history

    # Build lower triangular from network output (also calculate determinant)
    def lower_triangular_matrix(self, σ):
        Σ = []
        det = []
        start = 0
        end = 1
        for i in range(self.D):
            exp_val = tf.exp(σ[:, :, end-1])
            det.append(exp_val)
            if i > 0:
                Σ.append(tf.pad(tf.concat([σ[:, :, start:end-1], tf.expand_dims(exp_val, -1)], -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
            else:
                Σ.append(tf.pad(tf.expand_dims(exp_val, -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
            start = end
            end += i + 2
        Σ = tf.transpose(tf.stack(Σ), (1, 2, 0, 3))
        det = tf.reduce_prod(tf.stack(det), 0)
        return Σ, det

    # Split network output into means, covariances and weights (also returns determinant of covariance)
    def mapping(self, parameters):
        μ, Σ, α = tf.split(parameters, [self.M * self.D, self.M * self.D * (self.D + 1) // 2, self.M], 1)
        μ = tf.reshape(μ, (-1, self.M, self.D))
        Σ, det = self.lower_triangular_matrix(tf.reshape(Σ, (-1, self.M, self.D * (self.D + 1) // 2)))
        α = tf.nn.softmax(α)
        return μ, Σ, α, det

    #def log_sum_exp(self, x, axis=None):
    #    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    #    return tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis, keepdims=True)) + x_max

    # Build TensorFlow network
    def build_network(self):
        tf.reset_default_graph()
        # Set up a scope so this network is safe to update independently of an IMNN compression network
        with tf.variable_scope("MDN") as scope:
            self.parameter = tf.placeholder(tf.float32, shape = (None, self.D), name = "parameter")
            self.true = tf.placeholder(tf.float32, shape = (None, self.D), name = "true")
            self.ϵ = tf.placeholder(tf.float32, shape = (), name = "epsilon")
            self.layers = [self.parameter]
            self.weights = []
            self.biases = []
            for i in range(len(self.n_hidden)):
                with tf.variable_scope('layer_' + str(i + 1)):
                    if i == 0:
                        self.weights.append(tf.get_variable("weights", [self.D, self.n_hidden[i]], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.D))))
                        self.biases.append(tf.get_variable("biases", [self.n_hidden[i]], initializer = tf.constant_initializer(0.0)))
                    elif i == len(self.n_hidden) - 1:
                        self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.N], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.n_hidden[i]))))
                        self.biases.append(tf.get_variable("biases", [self.N], initializer = tf.constant_initializer(0.0)))
                    else:
                        self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.n_hidden[i + 1]], initializer = tf.random_normal_initializer(0., np.sqrt(2/self.n_hidden[i]))))
                        self.biases.append(tf.get_variable("biases", [self.n_hidden[i + 1]], initializer = tf.constant_initializer(0.0)))
                if i < len(self.n_hidden) - 1:
                    self.layers.append(self.activations[i](tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1])))
                else:
                    self.layers.append(tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1]))
            self.μ, self.Σ, self.α, self.det = self.mapping(self.layers[-1])
            self.μ = tf.identity(self.μ, name = "mu")
            self.Σ = tf.identity(self.Σ, name = "Sigma")
            self.α = tf.identity(self.α, name = "alpha")
            self.det = tf.identity(self.μ, name = "det")
            self.like = tf.reduce_sum(tf.exp(-0.5*tf.reduce_sum(tf.square(tf.einsum("ijlk,ijk->ijl", self.Σ, tf.subtract(tf.expand_dims(self.true, 1), self.μ))), 2) + tf.log(self.α) + tf.log(self.det) - self.D*np.log(2. * np.pi) / 2.), 1, name = "like")
            self.neg_log_normal_mixture_likelihood = -tf.reduce_mean(tf.log(self.like + self.ϵ), name = "neg_log_normal_mixture_likelihood")
            self.train = tf.train.AdamOptimizer(self.η).minimize(self.neg_log_normal_mixture_likelihood)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
