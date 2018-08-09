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

    def __init__(self, simulator, prior, asymptotic_posterior, Finv, theta_fiducial, data, n_components, simulator_args = None, n_hidden = [50, 50], activations = ['tanh', 'tanh'], names=None, labels=None, ranges=None, nwalkers=100, posterior_chain_length=1000, proposal_chain_length=100):

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
    
        # Training loss, validation loss
        self.loss = []
        self.val_loss = []
        self.loss_trace = []
        self.val_loss_trace = []
        self.n_sim_trace = []
    
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
    
        data_samples = np.zeros((n_batch, self.npar))
        parameter_samples = np.zeros((n_batch, self.npar))

        # Run simulations, catching exceptions (when simulator returns np.nan)
        i = 0
        count = 0
        while count < n_batch:
            try:
                sim = self.simulator(ps[i,:], self.simulator_args)
                if sum(np.isnan(sim.flatten())) == 0:
                    data_samples[count,:] = sim
                    parameter_samples[count,:] = ps[i,:]
                    count += 1
                else:
                    print("Simulator return NaNs with parameter values: {}".format(ps[i,:]))
                i += 1
            except:
                print("Simulator return exception with parameter values: {}".format(ps[i,:]))
                i += 1
    
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

        # Generate initial theta values from some broad proposal
        ps = np.array([proposal.draw() for i in range(5*n_initial)])

        # Run simulations at those theta values
        print('Running initial {} sims...'.format(n_initial))
        self.xs, self.ps = self.run_simulation_batch(n_initial, ps)
        print('Done.')

        # Construct the initial training-set
        self.ps = (self.ps - self.theta_fiducial)/self.fisher_errors
        self.xs = (self.xs - self.theta_fiducial)/self.fisher_errors
        self.x_train = self.ps.astype(np.float32)
        self.y_train = self.xs.astype(np.float32)
        self.n_sims = len(self.x_train)

        # Train the network on these initial simulations
        history = self.mdn.fit(self.x_train, self.y_train,
                    batch_size=int(self.n_sims/8), epochs=epochs, verbose=1, validation_split=validation_split, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')])
                    
        # Update the loss and validation loss
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.loss_trace.append(history.history['loss'][-1])
        self.val_loss_trace.append(history.history['val_loss'][-1])
        self.n_sim_trace.append(self.n_sims)
        
        # Generate posterior samples
        print('Sampling approximate posterior...')
        self.posterior_samples = self.emcee_sample(self.log_posterior, [self.posterior_samples[-i,:] for i in range(self.nwalkers)], main_chain=self.posterior_chain_length)
        print('Done.')

        # If plot == True, plot the current posterior estimate
        if plot == True:
            self.triangle_plot([self.posterior_samples])

        # Loop through a number of populations
        for i in range(n_populations):
            
            # Current population
            print('Population {}/{}'.format(i+1, n_populations))
    
            # Sample the current posterior approximation
            print('Sampling proposal density...')
            self.proposal_samples = self.emcee_sample(self.log_geometric_mean_proposal, [self.proposal_samples[-i,:] for i in range(self.nwalkers)], main_chain=self.proposal_chain_length)
            print('Done.')
    
            # Run simulations
            print('Running {} sims...'.format(n_batch))
            xs_batch, ps_batch = self.run_simulation_batch(n_batch, self.proposal_samples)
            print('Done.')
    
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
                   batch_size=int(self.n_sims/8), epochs=epochs, verbose=1, validation_split=validation_split, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')])
                   
            # Update the loss and validation loss
            self.loss = np.concatenate([self.loss, history.history['loss']])
            self.val_loss = np.concatenate([self.val_loss, history.history['val_loss']])
            self.loss_trace.append(history.history['loss'][-1])
            self.val_loss_trace.append(history.history['val_loss'][-1])
            self.n_sim_trace.append(self.n_sims)

            # Generate posterior samples
            print('Sampling approximate posterior...')
            self.posterior_samples = self.emcee_sample(self.log_posterior, [self.posterior_samples[i] for i in range(self.nwalkers)], main_chain=self.posterior_chain_length)
            print('Done.')

            # If plot == True
            if plot == True:
                self.triangle_plot([self.posterior_samples])

        # Train the network over some more epochs
        print('Final round of training with larger SGD batch size...')
        self.mdn.fit(self.x_train, self.y_train,
          batch_size=self.n_sims, epochs=300, verbose=1, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')])
        print('Done.')

        print('Sampling approximate posterior...')
        self.posterior_samples = self.emcee_sample(self.log_posterior, [self.posterior_samples[-i,:] for i in range(self.nwalkers)], main_chain=self.posterior_chain_length)
        print('Done.')

        # if plot == True
        if plot == True:
            self.triangle_plot([self.posterior_samples])
    
    def fisher_pretraining(self, n_batch, proposal, plot=True, batch_size=100, validation_split=0.1, epochs=100, patience=20):

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
            self.triangle_plot([self.posterior_samples])

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
        
        if savefig == True:
            plt.savefig(filename)
        plt.show()

