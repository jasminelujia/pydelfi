import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Lambda
import getdist
from getdist import plots, MCSamples
import emcee
import matplotlib.pyplot as plt
#from tqdm import tnrange

class DelfiMixtureDensityNetwork():

    def __init__(self, simulator, prior, asymptotic_posterior, \
                 Finv, theta_fiducial, data, n_components, \
                 simulator_args = None, n_hidden = [50, 50], \
                 activations = ["tanh", "tanh"], names=None, \
                 labels=None, ranges=None, nwalkers=100, \
                 posterior_chain_length=1000, proposal_chain_length=100, \
                 rank=0, n_procs=1, comm=None, red_op=None, \
                 show_plot=True):

        # Input x and output t dimensions
        self.D = len(data)
        self.npar = len(theta_fiducial)

        # Number of GMM components
        self.M = n_components

        # Number of hiden units and activations
        self.n_hidden = n_hidden
        self.activations = activations

        # Total number of outputs for the neural network
        self.N = int((self.D + self.D * (self.D + 1) / 2 + 1)*self.M)

        # Constant used in loss function
        self.Dlog2pio2 = self.D*np.log(2. * np.pi) / 2.
        self.ϕ = 0#1e19
        self. ϵ = 1e-30

        ## Build TensorFlow model
        ##tf.reset_default_graph()
        #self.parameter = tf.placeholder(tf.float32, shape = (None, self.D), name = "parameters")
        #self.true = tf.placeholder(tf.float32, shape = (None, self.D), name = "true")
        #self.layers = [self.parameter]
        #self.weights = []
        #self.biases = []
        #for i in range(len(self.n_hidden)):
        #    with tf.variable_scope('layer_' + str(i + 1)):
        #        if i == 0:
        #            self.weights.append(tf.get_variable("weights", [self.D, self.n_hidden[i]], initializer = tf.random_normal_initializer(0., 1.)))
        #            self.biases.append(tf.get_variable("biases", [self.n_hidden[i]], initializer = tf.constant_initializer(0.1)))
        #        elif i == len(self.n_hidden) - 1:
        #            self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.N], initializer = tf.random_normal_initializer(0., 1.)))
        #            self.biases.append(tf.get_variable("biases", [self.N], initializer = tf.constant_initializer(0.1)))
        #        else:
        #            self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.n_hidden[i + 1]], initializer = tf.random_normal_initializer(0., 1.)))
        #            self.biases.append(tf.get_variable("biases", [self.n_hidden[i + 1]], initializer = tf.constant_initializer(0.1)))
        #    if i < len(self.n_hidden) - 1:
        #        self.layers.append(self.activations[i](tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1])))
        #    else:
        #        self.layers.append(tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1]))
        #self.μ, self.Σ, self.α, self.det = self.mapping(self.layers[-1])
        #self.diff = tf.expand_dims(self.true, 1) - self.μ
        #calc_loss = tf.exp(-0.5 * tf.einsum("ijlk,ijk->ij", self.Σ, tf.square(self.diff)) + tf.log(self.α) + tf.log(self.det) - self.Dlog2pio2)
        #calc_loss = tf.where(tf.is_nan(calc_loss), tf.zeros_like(calc_loss), calc_loss)
        #self.loss = -tf.reduce_mean(tf.log(tf.reduce_sum(tf.where(tf.is_inf(calc_loss), tf.zeros_like(calc_loss), calc_loss), 1) + 1e-19))
        #self.train = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())


        # Initialize the sequential Keras model
        self.mdn = Sequential()

        # Add the (dense) hidden layers
        for i in range(len(self.n_hidden)):
            self.mdn.add(Dense(self.n_hidden[i], activation=self.activations[i], input_shape=(self.D,)))

        # Linear output layer
        self.mdn.add(Dense(self.N, activation='linear'))

        # Compile the Keras model
        self.mdn.summary()
        self.mdn.compile(loss=self.neg_log_normal_mixture_likelihood,
                         optimizer='adam')

        # Prior and asymptotic posterior
        self.prior = prior
        self.asymptotic_posterior = asymptotic_posterior

        # Training data
        self.ps = []
        self.xs = []
        self.x_train = []
        self.y_train = []
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
        while i_acpt <= self.inds_acpt[-1] and i_prop <= self.inds_prop[-1]:
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
        # Predict means, covariances and weights given data
        y_out = self.mdn.predict(theta.astype(np.float32).reshape((1, self.D)))

        # Extract means, covariances and weights
        μ = y_out[0, :self.D*self.M].reshape((self.M, self.D))
        Σ, det = self.covariance_matrix(y_out[0, self.D*self.M:self.M*(self.D*(self.D + 1) // 2 + self.D)].reshape((1, self.M, self.D*(self.D + 1) // 2)), use_tf = False)
        α = np.exp(y_out[0, self.M*(self.D*(self.D + 1) // 2 + self.D):]) / np.sum(np.exp((y_out[0, self.M*(self.D*(self.D + 1) // 2 + self.D):])))

        # Calculate likelihood
        diff = (self.data - self.theta_fiducial)/self.fisher_errors-μ
        like = np.exp(-0.5 *np.einsum("ikj,ik->i", Σ, diff**2.) + np.log(α) + np.log(det) - self.Dlog2pio2)
        like[np.isnan(like)] = 0
        like[np.isinf(like)] = self.ϕ
        like = np.sum(like)
        #like = np.sum(α * np.exp(-0.5*np.einsum("ijk,ik->i", Σ, diff**2.)) * det / np.sqrt((2 * np.pi) ** self.D))

        if np.isnan(np.log(like)) == True:
            return -1e300
        else:
            return np.log(like)

    def sequential_training(self, n_initial, n_batch, n_populations, proposal, \
                            safety = 5, plot = True, batch_size=100, \
                            validation_split=0.1, epochs=100, patience=20):

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
            self.x_train = self.ps.astype(np.float32)
            self.y_train = self.xs.astype(np.float32)
            self.n_sims = len(self.x_train)

            # Train the network on these initial simulations
            kcb = keras.callbacks.EarlyStopping(monitor='val_loss', \
                                                min_delta=0, \
                                                patience=patience, \
                                                verbose=0, mode='auto')
            history = self.mdn.fit(self.x_train, self.y_train, \
                                   batch_size=batch_size, \
                                   epochs=epochs, verbose=1, \
                                   validation_split=validation_split, \
                                   callbacks=[kcb])

            # Update the loss and validation loss
            self.loss = history.history['loss']
            self.val_loss = history.history['val_loss']
            self.loss_trace.append(history.history['loss'][-1])
            self.val_loss_trace.append(history.history['val_loss'][-1])
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
                self.x_train = self.ps.astype(np.float32)
                self.y_train = self.xs.astype(np.float32)

                # Train the network on these initial simulations
                history = self.mdn.fit(self.x_train, self.y_train,
                                       batch_size=batch_size, \
                                       epochs=epochs, verbose=1, \
                                       validation_split=validation_split, \
                                       callbacks=[kcb])

                # Update the loss and validation loss
                self.loss = np.concatenate([self.loss, history.history['loss']])
                self.val_loss = np.concatenate([self.val_loss, history.history['val_loss']])
                self.loss_trace.append(history.history['loss'][-1])
                self.val_loss_trace.append(history.history['val_loss'][-1])
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

            # Train the network over some more epochs
            print('Final round of training with larger SGD batch size...')
            self.mdn.fit(self.x_train, self.y_train, \
                         batch_size=self.n_sims, epochs=300, \
                         verbose=1, validation_split=0.1, \
                         callbacks=[kcb])
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

    def fisher_pretraining(self, n_batch, proposal, plot=True, batch_size=100, validation_split=0.1, epochs=100, patience=20):

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
            #ps = np.zeros((n_batch + int(validation_split * n_batch), self.npar))
            #for i in range(0, n_batch + int(validation_split * n_batch)):
            #    ps[i,:] = (proposal.draw() - self.theta_fiducial)/self.fisher_errors

            # Sample data assuming a Gaussian likelihood
            xs = np.array([pss + np.dot(Ldd, np.random.normal(0, 1, self.npar)) for pss in ps])

            # Construct the initial training-set
            self.x_train = ps.astype(np.float32).reshape((n_batch, self.npar))
            self.y_train = xs.astype(np.float32).reshape((n_batch, self.npar))
            #self.x_train = ps[:n_batch].astype(np.float32).reshape((n_batch // batch_size, batch_size, self.npar))
            #self.y_train = xs[:n_batch].astype(np.float32).reshape((n_batch // batch_size, batch_size, self.npar))
            #self.x_test = ps[n_batch:].astype(np.float32).reshape((int(n_batch*(validation_split)) // batch_size, batch_size, self.npar))
            #self.y_test = xs[n_batch:].astype(np.float32).reshape((int(n_batch*(validation_split)) // batch_size, batch_size, self.npar))

            # Train network on initial (asymptotic) simulations
            print("Training on the pre-training data...")
            #tr_epoch = tnrange(epochs, desc = "Epochs")
            #for epoch in tr_epoch:
            #    tr_batch = tnrange(1)#tnrange(n_batch // batch_size, desc = "Batches")
            #    for batch in tr_batch:
            #        _, loss = self.sess.run([self.train, self.loss], feed_dict = {self.parameter: self.x_train[batch], self.true: self.y_train[batch]})
            #        tr_batch.set_postfix(loss=loss)

            history = self.mdn.fit(self.x_train, self.y_train,
              batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')])
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
            self.loss_trace.append(history.history['loss'][-1])
            self.val_loss_trace.append(history.history['val_loss'][-1])
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

    # Build lower triangular from network output (also calculate determinant)
    def covariance_matrix(self, σ, use_tf = True):
        Σ = []
        start = 0
        end = 1
        for i in range(self.D):
            if use_tf:
                exp_val = tf.exp(σ[:, :, end-1])
            else:
                exp_val = np.exp(σ[:, :, end-1])
            if i > 0:
                det *= exp_val
                if use_tf:
                    Σ.append(tf.pad(tf.concat([σ[:, :, start:end-1], tf.expand_dims(exp_val, -1)], -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
                else:
                    Σ.append(np.pad(np.concatenate([σ[:, :, start:end-1], exp_val[:, :, np.newaxis]], -1), [[0, 0], [0, 0], [0, self.D-i-1]], "constant"))
            else:
                det = exp_val
                if use_tf:
                    Σ.append(tf.pad(tf.expand_dims(exp_val, -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
                else:
                    Σ.append(np.pad(exp_val[:, :, np.newaxis], [[0, 0], [0, 0], [0, self.D-i-1]], "constant"))
            start = end
            end += i + 2
        if use_tf:
            Σ = tf.transpose(tf.stack(Σ), (1, 2, 0, 3))
        else:
            Σ = np.swapaxes(np.swapaxes(Σ, 0, 1), 1, 2)[0]
            det = det[0]
        return Σ, det

    # Split network output into means, covariances and weights (also returns determinant of covariance)
    def mapping(self, parameters):
        μ, Σ, α = tf.split(parameters, [self.M * self.D, self.M * self.D * (self.D + 1) // 2, self.M], 1)
        μ = tf.reshape(μ, (-1, self.M, self.D))
        Σ, det = self.covariance_matrix(tf.reshape(Σ, (-1, self.M, self.D * (self.D + 1) // 2)))
        α = tf.nn.softmax(α)
        return μ, Σ, α, det

    # Loss function
    def neg_log_normal_mixture_likelihood(self, true, parameters):
        μ, Σ, α, det = self.mapping(parameters)
        diff = tf.expand_dims(true, 1) - μ
        calc_loss = tf.exp(-0.5 * tf.einsum("ijlk,ijk->ij", Σ, tf.square(diff)) + tf.log(α) + tf.log(det) - self.Dlog2pio2)
        calc_loss = tf.where(tf.is_nan(calc_loss), tf.zeros_like(calc_loss), calc_loss)
        return -tf.reduce_mean(tf.log(tf.reduce_sum(tf.where(tf.is_inf(calc_loss), tf.ones_like(calc_loss) * self.ϕ, calc_loss), 1) + self.ϵ))

    def log_sum_exp(self, x, axis=None):
        x_max = tf.reduce_max(x, axis=axis, keepdims=True)
        return tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis, keepdims=True)) + x_max
