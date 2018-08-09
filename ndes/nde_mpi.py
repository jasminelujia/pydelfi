import theano
import theano.tensor as T
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Lambda
import getdist
from getdist import plots, MCSamples
from ndes.losses import *
import keras
import emcee
import matplotlib.pyplot as plt

class DelfiMixtureDensityNetwork():

    def __init__(self, simulator, prior, asymptotic_posterior, \
                 Finv, theta_fiducial, data, n_components, \
                 n_hidden = [50, 50], activations = ['tanh', 'tanh'], \
                 names=None, labels=None, ranges=None, nwalkers=100, \
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
        self.N = (self.D + self.D**2 + 1)*self.M
    
        # Initialize the sequential Keras model
        self.mdn = Sequential()
        
        # Add the (dense) hidden layers
        for i in range(len(self.n_hidden)):
            self.mdn.add(Dense(self.n_hidden[i], activation=self.activations[i], input_shape=(self.D,)))
        
        # Linear output layer
        self.mdn.add(Dense(self.N, activation='linear'))
        
        # Compile the Keras model
        self.mdn.compile(loss=neg_log_normal_mixture_likelihood,
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
            n_j_to_allocate = n_j_remain / n_p_remain
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
        
        # Run samples assigned to each process
        data_samples = np.zeros((n_batch, self.npar))
        for i in self.job_list:
            data_samples[i,:] = self.simulator(ps[i,:])

        # Reduce results from all processes
        data_samples = self.complete_array(data_samples)
    
        return data_samples

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
    
        y_out = self.mdn.predict(theta.astype(np.float32).reshape((1, self.D)))
    
        means = y_out[0, : self.D*self.M].reshape((self.M, self.D))
        sigmas = y_out[0, self.D*self.M: self.D*self.M + self.M*self.D*self.D].reshape((self.M, self.D, self.D))
        weights = np.exp(y_out[0, self.D*self.M + self.M*self.D*self.D:])/sum(np.exp(y_out[0, self.D*self.M + self.M*self.D*self.D:]))
        like = 0
        for i in range(self.M):
            L = np.tril(sigmas[i,:,:], k=-1) + np.diag(np.exp(np.diagonal(sigmas[i,:,:])))
            like += weights[i]*np.exp(-0.5*np.sum(np.dot(L.T, ((self.data - self.theta_fiducial)/self.fisher_errors - means[i,:]))**2))*np.prod(np.diag(L))/np.sqrt((2*np.pi)**self.D)
    
        if np.isnan(np.log(like)) == True:
            return -1e300
        else:
            return np.log(like)

    def sequential_training(self, n_initial, n_batch, n_populations, proposal, plot = True, batch_size=100, validation_split=0.1, epochs=100, patience=20):

        # Generate initial theta values from some broad proposal on 
        # master process and share with other processes. Allocate jobs.
        if self.rank == 0:
            ps = np.array([proposal.draw() for i in range(n_initial)])
        else:
            ps = np.zeros((n_initial, self.npar))
        if self.use_mpi:
            self.comm.Bcast(ps, root=0)
        self.ps = ps
        self.job_list = self.allocate_jobs(n_initial)

        # Run simulations at those theta values
        if self.rank == 0:
            print('Running initial {} sims...'.format(n_initial))
        self.xs = self.run_simulation_batch(n_initial, self.ps)
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
            # other processes
            if self.rank == 0:

                # Current population
                print('Population {}/{}'.format(i+1, n_populations))
        
                # Sample the current posterior approximation
                print('Sampling proposal density...')
                self.proposal_samples = \
                    self.emcee_sample(self.log_geometric_mean_proposal, \
                                      [self.proposal_samples[-j,:] for j in range(self.nwalkers)], \
                                      main_chain=self.proposal_chain_length)
                ps_batch = self.proposal_samples[-n_batch:,:]
                print('Done.')

            else:
                ps_batch = np.zeros((n_batch, self.npar))
            if self.use_mpi:
                self.comm.Bcast(ps_batch, root=0)

            # Run simulations
            if self.rank == 0:
                print('Running {} sims...'.format(n_batch))
            xs_batch = self.run_simulation_batch(n_batch, ps_batch)
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

            # Construct the initial training-set
            self.x_train = ps.astype(np.float32).reshape((n_batch, self.npar))
            self.y_train = xs.astype(np.float32).reshape((n_batch, self.npar))

            # Train network on initial (asymptotic) simulations
            print("Training on the pre-training data...")
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

