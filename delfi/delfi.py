import tensorflow as tf
import getdist
from getdist import plots, MCSamples
import ndes.ndes
import ndes.train
import emcee
import matplotlib.pyplot as plt
import distributions.priors as priors
import numpy as np
import tqdm
import scipy.optimize as optimization

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

class Delfi():

    def __init__(self, data, prior, nde, \
                 Finv= None, theta_fiducial = None, param_limits = None, param_names=None, nwalkers=100, \
                 posterior_chain_length=1000, proposal_chain_length=100, \
                 rank=0, n_procs=1, comm=None, red_op=None, \
                 show_plot=True, results_dir = "", progress_bar=True, input_normalization = None):
        
        # Data
        self.data = data
        self.D = len(data)
        
        # Prior
        self.prior = prior

        # Number of parameters
        self.npar = nde[0].n_inputs

        # Initialize the NDEs, trainers, and stacking weights (for stacked density estimators)
        self.n_ndes = len(nde)
        self.nde = nde
        self.trainer = [ndes.train.ConditionalTrainer(nde[i]) for i in range(self.n_ndes)]
        self.stacking_weights = np.zeros(self.n_ndes)

        # Tensorflow session for the NDE training
        self.sess = tf.Session(config = tf.ConfigProto())
        self.sess.run(tf.global_variables_initializer())
        
        # Parameter limits
        if param_limits is not None:
            # Set to provided prior limits if provided
            self.lower = param_limits[0]
            self.upper = param_limits[1]
        else:
            # Else set to max and min float32
            self.lower = np.ones(self.npar)*np.finfo(np.float32).min
            self.upper = np.ones(self.npar)*np.finfo(np.float32).max

        # Fisher matrix and fiducial parameters
        if Finv is not None:
            self.Finv = Finv
            self.fisher_errors = np.sqrt(np.diag(self.Finv))
            self.theta_fiducial = theta_fiducial
            self.asymptotic_posterior = priors.TruncatedGaussian(self.data, self.Finv, self.lower, self.upper)
        else:
            self.Finv = None
            self.fisher_errors = None
            self.theta_fiducial = None
            self.asymptotic_posterior = None

        # Re-scaling for inputs to NDE
        self.input_normalization = input_normalization
        if input_normalization is None:
            self.x_mean = np.zeros(self.D)
            self.x_std = np.ones(self.D)
            self.p_mean = np.zeros(self.npar)
            self.p_std = np.ones(self.npar)
        elif input_normalization is 'fisher':
            self.x_mean = self.theta_fiducial
            self.x_std = self.fisher_errors
            self.p_mean = self.theta_fiducial
            self.p_std = self.fisher_errors
        else:
            self.x_mean, self.x_std, self.p_mean, self.p_std = input_normalization

        # Training data [initialize empty]
        self.ps = np.array([]).reshape(0,self.npar)
        self.xs = np.array([]).reshape(0,self.D)
        self.x_train = tf.placeholder(tf.float32, shape = (None, self.D))
        self.y_train = tf.placeholder(tf.float32, shape = (None, self.D))
        self.n_sims = 0
        
        # MCMC chain parameters
        self.nwalkers = nwalkers
        self.posterior_chain_length = posterior_chain_length
        self.proposal_chain_length = proposal_chain_length
        
        # MCMC samples of learned posterior
        if self.asymptotic_posterior is not None:
            self.posterior_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.posterior_chain_length)])
            self.proposal_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.proposal_chain_length)])
        else:
            self.posterior_samples = np.array([self.prior.draw() for i in range(self.nwalkers*self.posterior_chain_length)])
            self.proposal_samples = np.array([self.prior.draw() for i in range(self.nwalkers*self.proposal_chain_length)])
    
        # Parameter names and ranges for plotting with GetDist
        self.names = param_names
        self.labels = param_names
        self.ranges = dict(zip(param_names, [ [self.lower[i], self.upper[i]] for i in range(self.npar) ]))
        self.show_plot = show_plot
        
        # Results directory
        self.results_dir = results_dir
        
        # Training loss, validation loss
        self.training_loss = [np.array([]) for i in range(self.n_ndes)]
        self.validation_loss = [np.array([]) for i in range(self.n_ndes)]
        self.stacked_sequential_training_loss = []
        self.stacked_sequential_validation_loss = []
        self.sequential_nsims = []

        # MPI-specific setup
        self.rank = rank
        self.n_procs = n_procs
        if n_procs > 1:
            self.use_mpi = True
            self.comm = comm
            self.red_op = red_op
        else:
            self.use_mpi = False

        # Are we in a jupyter notebook or not?
        self.nb = isnotebook()

        # Show progress bars?
        self.progress_bar = progress_bar
    
    # Divide list of jobs between MPI processes
    def allocate_jobs(self, n_jobs):
        n_j_allocated = 0
        for i in range(self.n_procs):
            n_j_remain = n_jobs - n_j_allocated
            n_p_remain = self.n_procs - i
            n_j_to_allocate = int(n_j_remain / n_p_remain)
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
    
    # NDE log likelihood (individual NDE)
    def log_likelihood_individual(self, i, theta):
    
        lnL = self.nde[i].eval((np.atleast_2d((theta-self.p_mean)/self.p_std), np.atleast_2d((self.data-self.x_mean)/self.x_std)), self.sess)
    
        if np.isnan(lnL) == True:
            return -1e300
        else:
            return lnL

    # NDE log likelihood (stacked)
    def log_likelihood_stacked(self, theta):

        # Stack the likelihoods
        L = 0
        for n in range(self.n_ndes):
            L += self.stacking_weights[n]*np.exp(self.nde[n].eval((np.atleast_2d((theta-self.p_mean)/self.p_std), np.atleast_2d((self.data-self.x_mean)/self.x_std)), self.sess))
        lnL = np.log(L)
        if np.isnan(lnL) == True:
            return -1e300
        else:
            return lnL

    # Log posterior
    def log_posterior_stacked(self, x):
        
        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return self.log_likelihood_stacked(x) + self.prior.logpdf(x)

    # Log posterior
    def log_posterior_individual(self, i, x):
        
        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return self.log_likelihood_individual(i, x) + self.prior.logpdf(x)
    
    # Log posterior
    def log_geometric_mean_proposal_stacked(self, x):
        
        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return 0.5 * (self.log_likelihood_stacked(x) + 2 * self.prior.logpdf(x) )

    # Log posterior
    def log_geometric_mean_proposal_individual(self, i, x):
        
        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return 0.5 * (self.log_likelihood_individual(i, x) + 2 * self.prior.logpdf(x) )

    # Bayesian optimization acquisition function
    def acquisition(self, theta):

        # Compute log_posteriors
        Ls = np.array([self.log_posterior_individual(i, theta) for i in range(self.n_ndes)])
    
        # Check whether prior is zero or not
        if self.prior.pdf(theta) == 0:
            return 0
        else:
            return self.log_posterior_stacked(theta)*np.sqrt(np.average((Ls - np.average(Ls, weights = self.stacking_weights, axis=0))**2, weights=self.stacking_weights, axis=0))
                
    # Bayesian optimization training
    def bayesian_optimization_training(self, simulator, compressor, n_batch, n_populations, n_optimizations = 10, \
                                       simulator_args = None, compressor_args = None, plot = False, batch_size = 100, \
                                       validation_split = 0.1, epochs = 300, patience = 20, seed_generator = None, \
                                       save_intermediate_posteriors = False, sub_batch = 1):
    
        # Loop over n_populations
        for i in range(n_populations):
    
            # Find acquisition point...
            print('Finding optimal acquisition point...')
            A_optimal = 0
            theta_optimal = self.theta_fiducial
            for i in range(n_optimizations):
                res = optimization.basinhopping(lambda x: -self.acquisition(x), x0=self.theta_fiducial)
                if res.fun < A_optimal:
                    A_optimal = res.fun
                    theta_optimal = res.x
        
            # Array of parameters to run simulations
            ps = np.array([theta_optimal for k in range(n_batch)])
            
            # Run a small batch of simulations at the acquisition point
            xs_batch, ps_batch = self.run_simulation_batch(n_batch, ps, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)
            
            # Augment the training data
            self.add_simulations(xs_batch, ps_batch)
            
            # Re-train the networks
            self.train_ndes(training_data=[self.x_train, self.y_train], batch_size=max(self.n_sims//8, batch_size), validation_split=validation_split, epochs=epochs, patience=patience, saver_name=None)

            # Save the losses
            self.stacked_sequential_training_loss.append(np.sum(np.array([self.training_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
            self.stacked_sequential_validation_loss.append(np.sum(np.array([self.validation_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
            self.sequential_nsims.append(self.n_sims)
    
    # Run n_batch simulations
    def run_simulation_batch(self, n_batch, ps, simulator, compressor, simulator_args, compressor_args, seed_generator = None, sub_batch = 1):
        
        # Random seed generator: set to unsigned 32 bit int random numbers as default
        if seed_generator is None:
            seed_generator = lambda: np.random.randint(2147483647)
        
        # Dimension outputs
        data_samples = np.zeros((n_batch*sub_batch, self.npar))
        parameter_samples = np.zeros((n_batch*sub_batch, self.npar))
        
        # Run samples assigned to each process, catching exceptions
        # (when simulator returns np.nan).
        i_prop = self.inds_prop[0]
        i_acpt = self.inds_acpt[0]
        err_msg = 'Simulator returns {:s} for parameter values: {} (rank {:d})'
        if self.progress_bar:
            if self.nb:
                pbar = tqdm.tqdm_notebook(total = self.inds_acpt[-1], desc = "Simulations")
            else:
                pbar = tqdm.tqdm(total = self.inds_acpt[-1], desc = "Simulations")
        while i_acpt <= self.inds_acpt[-1]:
            try:
                sims = simulator(ps[i_prop,:], seed_generator(), simulator_args, sub_batch)
                
                # Make sure the sims are the right shape
                if sub_batch == 1 and len(sims) != 1:
                    sims = np.array([sims])
                compressed_sims = np.array([compressor(sims[k], compressor_args) for k in range(sub_batch)])
                if np.all(np.isfinite(compressed_sims.flatten())):
                    data_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = compressed_sims
                    parameter_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = ps[i_prop,:]
                    i_acpt += 1
                    if self.progress_bar:
                        pbar.update(1)
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

    def sequential_training(self, simulator, compressor, n_initial, n_batch, n_populations, proposal = None, \
                            simulator_args = None, compressor_args = None, safety = 5, plot = True, batch_size = 100, \
                            validation_split = 0.1, epochs = 300, patience = 20, seed_generator = None, \
                            save_intermediate_posteriors = True, sub_batch = 1):

        # Set up the initial parameter proposal density
        if proposal is None:
            if self.input_normalization is 'fisher':
                proposal = priors.TruncatedGaussian(self.data, 9*self.Finv, self.lower, self.upper)
            elif self.Finv is not None:
                proposal = priors.TruncatedGaussian(self.theta_fiducial, 9*self.Finv, self.lower, self.upper)
            else:
                proposal = self.prior
                    
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
        xs_batch, ps_batch = self.run_simulation_batch(n_initial, ps, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

        # Train on master only
        if self.rank == 0:

            # Construct the initial training-set
            self.load_simulations(xs_batch, ps_batch)

            # Train the network on these initial simulations
            self.train_ndes(training_data=[self.x_train, self.y_train], batch_size=max(self.n_sims//8, batch_size), validation_split=validation_split, epochs=epochs, patience=patience, saver_name=None)
            self.stacked_sequential_training_loss.append(np.sum(np.array([self.training_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
            self.stacked_sequential_validation_loss.append(np.sum(np.array([self.validation_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
            self.sequential_nsims.append(self.n_sims)
            
            # Generate posterior samples
            if save_intermediate_posteriors:
                print('Sampling approximate posterior...')
                self.posterior_samples = self.emcee_sample(self.log_posterior_stacked, \
                                  [self.posterior_samples[-i,:] for i in range(self.nwalkers)], \
                                  main_chain=self.posterior_chain_length)
            
                # Save posterior samples to file
                f = open('{}posterior_samples_0.dat'.format(self.results_dir), 'w')
                np.savetxt(f, self.posterior_samples)
                f.close()
            
                print('Done.')

                # If plot == True, plot the current posterior estimate
                if plot == True:
                    self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='{}seq_train_post_0.pdf'.format(self.results_dir))

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
                    self.emcee_sample(self.log_geometric_mean_proposal_stacked, \
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
            xs_batch, ps_batch = self.run_simulation_batch(n_batch, ps_batch, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

            # Train on master only
            if self.rank == 0:
        
                # Augment the training data
                self.add_simulations(xs_batch, ps_batch)
        
                # Train the network on these initial simulations
                self.train_ndes(training_data=[self.x_train, self.y_train], batch_size=max(self.n_sims//8, batch_size), validation_split=0.1, epochs=epochs, patience=patience, saver_name=None)
                self.stacked_sequential_training_loss.append(np.sum(np.array([self.training_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
                self.stacked_sequential_validation_loss.append(np.sum(np.array([self.validation_loss[n][-1]*self.stacking_weights[n] for n in range(self.n_ndes)])))
                self.sequential_nsims.append(self.n_sims)

                # Generate posterior samples
                if save_intermediate_posteriors:
                    print('Sampling approximate posterior...')
                    self.posterior_samples = self.emcee_sample(self.log_posterior_stacked, \
                                      [self.posterior_samples[j] for j in range(self.nwalkers)], \
                                      main_chain=self.posterior_chain_length)
                
                    # Save posterior samples to file
                    f = open('{}posterior_samples_{:d}.dat'.format(self.results_dir, i+1), 'w')
                    np.savetxt(f, self.posterior_samples)
                    f.close()

                    print('Done.')

                    # If plot == True
                    if plot == True:
                        # Plot the posterior
                        self.triangle_plot([self.posterior_samples], \
                                        savefig=True, \
                                        filename='{}seq_train_post_{:d}.pdf'.format(self.results_dir, i + 1))

                # Plot training convergence
                if plot == True:
                    # Plot the training loss convergence
                    self.sequential_training_plot(savefig=True, filename='{}seq_train_loss.pdf'.format(self.results_dir))

    def train_ndes(self, training_data, batch_size=100, validation_split=0.1, epochs=500, patience=20, saver_name=None):
    
        # Train the networks
        for n in range(self.n_ndes):
            # Train the NDE
            val_loss, train_loss = self.trainer[n].train(self.sess, training_data, validation_split = validation_split, epochs=epochs, batch_size=batch_size, progress_bar=self.progress_bar, patience=patience, saver_name='{}tmp_model'.format(self.results_dir))
        
            # Save the training and validation losses
            self.training_loss[n] = np.concatenate([self.training_loss[n], train_loss])
            self.validation_loss[n] = np.concatenate([self.validation_loss[n], val_loss])

        # Update weights for stacked density estimator
        self.stacking_weights = np.exp(-np.array([self.training_loss[i][-1] for i in range(self.n_ndes)]))
        self.stacking_weights = self.stacking_weights/sum(self.stacking_weights)

    def load_simulations(self, xs_batch, ps_batch):
        
        # Set the input normalizations if None specified
        if self.input_normalization is None:
            self.p_mean = np.mean(ps_batch, axis = 0)
            self.p_std = np.std(ps_batch, axis = 0)
            self.x_mean = np.mean(xs_batch, axis = 0)
            self.x_std = np.std(xs_batch, axis = 0)

        ps_batch = (ps_batch - self.p_mean)/self.p_std
        xs_batch = (xs_batch - self.x_mean)/self.x_std
        self.ps = np.concatenate([self.ps, ps_batch])
        self.xs = np.concatenate([self.xs, xs_batch])
        self.x_train = self.ps.astype(np.float32)
        self.y_train = self.xs.astype(np.float32)
        self.n_sims += len(ps_batch)
    
    def add_simulations(self, xs_batch, ps_batch):
        
        ps_batch = (ps_batch - self.p_mean)/self.p_std
        xs_batch = (xs_batch - self.x_mean)/self.x_std
        self.ps = np.concatenate([self.ps, ps_batch])
        self.xs = np.concatenate([self.xs, xs_batch])
        self.x_train = self.ps.astype(np.float32)
        self.y_train = self.xs.astype(np.float32)
        self.n_sims += len(ps_batch)
    
    def fisher_pretraining(self, n_batch=50000, plot=True, batch_size=100, validation_split=0.1, epochs=300, patience=10):

        # Train on master only
        if self.rank == 0:

            # Generate fisher pre-training data
            
            # Broader proposal
            proposal = priors.TruncatedGaussian(self.theta_fiducial, 9*self.Finv, self.lower, self.upper)

            # Anticipated covariance of the re-scaled data
            Cdd = np.zeros((self.npar, self.npar))
            for i in range(self.npar):
                for j in range(self.npar):
                    Cdd[i,j] = self.Finv[i,j]/(self.fisher_errors[i]*self.fisher_errors[j])
            Ldd = np.linalg.cholesky(Cdd)
            Cddinv = np.linalg.inv(Cdd)
            ln2pidetCdd = np.log(2*np.pi*np.linalg.det(Cdd))
            
            # Sample parameters from some broad proposal
            ps = np.zeros((3*n_batch, self.npar))
            for i in range(0, n_batch):
                # Draws from prior
                ps[i,:] = (self.prior.draw() - self.theta_fiducial)/self.fisher_errors
                
                # Draws from asymptotic posterior
                ps[n_batch + i,:] = (self.asymptotic_posterior.draw() - self.theta_fiducial)/self.fisher_errors
                
                # Drawn from Gaussian with 3x anticipated covariance matrix
                ps[2*n_batch + i,:] = (proposal.draw() - self.theta_fiducial)/self.fisher_errors
            
            # Sample data assuming a Gaussian likelihood
            xs = np.array([pss + np.dot(Ldd, np.random.normal(0, 1, self.npar)) for pss in ps])
            
            # Construct the initial training-set
            fisher_x_train = ps.astype(np.float32).reshape((3*n_batch, self.npar))
            fisher_y_train = xs.astype(np.float32).reshape((3*n_batch, self.npar))
            
            # Train the networks on these initial simulations
            self.train_ndes(training_data=[fisher_x_train, fisher_y_train], validation_split = validation_split, epochs=epochs, batch_size=batch_size, patience=patience, saver_name=None)

    def triangle_plot(self, samples, savefig = False, filename = None):

        mc_samples = [MCSamples(samples=s, names = self.names, labels = self.labels, ranges = self.ranges) for s in samples]

        # Triangle plot
        g = plots.getSubplotPlotter(width_inch = 12)
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.6
        g.settings.axes_fontsize=14
        g.settings.legend_fontsize=16
        g.settings.lab_fontsize=20
        g.triangle_plot(mc_samples, filled_compare=True, normalized=True)
        for i in range(0, len(samples[0][0,:])):
            for j in range(0, i+1):
                ax = g.subplots[i,j]
                xtl = ax.get_xticklabels()
                ax.set_xticklabels(xtl, rotation=45)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        
        if savefig:
            plt.savefig(filename)
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def sequential_training_plot(self, savefig = False, filename = None):

        plt.close()
        columnwidth = 18 # cm
        aspect = 1.67
        pts_per_inch = 72.27
        inch_per_cm = 2.54
        width = columnwidth/inch_per_cm
        plt.rcParams.update({'figure.figsize': [width, width / aspect],
                          'backend': 'pdf',
                          'font.size': 15,
                          'legend.fontsize': 15,
                          'legend.frameon': False,
                          'legend.loc': 'best',
                          'lines.markersize': 3,
                          'lines.linewidth': .5,
                          'axes.linewidth': .5,
                          'axes.edgecolor': 'black'})
                          
        # Trace plot of the training and validation loss as a function of the number of simulations ran
        plt.scatter(self.sequential_nsims, self.stacked_sequential_training_loss, s = 20, alpha = 0.5, color = 'red')
        plt.plot(self.sequential_nsims, self.stacked_sequential_training_loss, color = 'red', lw = 2, alpha = 0.5, label = 'training loss')
        plt.scatter(self.sequential_nsims, self.stacked_sequential_validation_loss, s = 20, alpha = 0.5, color = 'blue')
        plt.plot(self.sequential_nsims, self.stacked_sequential_validation_loss, color = 'blue', lw = 2, alpha = 0.5, label = 'validation loss')

        plt.xlabel(r'number of simulations, $n_\mathrm{sims}$')
        plt.ylabel(r'negative log loss, $-\mathrm{ln}\,U$')
        plt.tight_layout()
        plt.legend()

        if savefig:
            plt.savefig(filename)
        if self.show_plot:
            plt.show()
        else:
            plt.close()




class DelfiSingle():

    def __init__(self, data, prior, nde, \
                 Finv, theta_fiducial, param_limits = None, param_names=None, nwalkers=100, \
                 posterior_chain_length=1000, proposal_chain_length=100, \
                 rank=0, n_procs=1, comm=None, red_op=None, \
                 show_plot=True, results_dir = "", progress_bar=True):
        
        # Data
        self.data = data
        self.D = len(data)
        
        # Prior
        self.prior = prior

        # Input x and output t dimensions
        self.npar = len(theta_fiducial)
    
        # Initialize the NDE and trainer
        self.nde = nde
        self.trainer = ndes.train.ConditionalTrainer(nde)

        # Tensorflow session for the NDE training
        self.sess = tf.Session(config = tf.ConfigProto())
        self.sess.run(tf.global_variables_initializer())

        # Fisher matrix and fiducial parameters
        self.Finv = Finv
        self.fisher_errors = np.sqrt(np.diag(self.Finv))
        self.theta_fiducial = theta_fiducial
        self.npar = len(theta_fiducial)
        
        # Re-scaling for inputs to NDE
        self.x_mean = x_mean
        self.x_std = x_std
        self.p_mean = p_mean
        self.p_std = p_std
        
        # Parameter limits
        if param_limits is not None:
            # Set to provided prior limits if provided
            self.lower = param_limits[0]
            self.upper = param_limits[1]
        else:
            # Else set to max and min float32
            self.lower = np.ones(self.npar)*np.finfo(np.float32).min
            self.upper = np.ones(self.npar)*np.finfo(np.float32).max
        
        # Asymptotic posterior
        self.asymptotic_posterior = priors.TruncatedGaussian(self.data, self.Finv, self.lower, self.upper)

        # Training data [initialize empty]
        self.ps = np.array([]).reshape(0,self.npar)
        self.xs = np.array([]).reshape(0,self.D)
        self.x_train = tf.placeholder(tf.float32, shape = (None, self.D))
        self.y_train = tf.placeholder(tf.float32, shape = (None, self.D))
        self.n_sims = 0
        
        # MCMC chain parameters
        self.nwalkers = nwalkers
        self.posterior_chain_length = posterior_chain_length
        self.proposal_chain_length = proposal_chain_length
        
        # MCMC samples of learned posterior
        self.posterior_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.posterior_chain_length)])
        self.proposal_samples = np.array([self.asymptotic_posterior.draw() for i in range(self.nwalkers*self.proposal_chain_length)])
    
        # Parameter names and ranges for plotting with GetDist
        self.names = param_names
        self.labels = param_names
        self.ranges = dict(zip(param_names, [ [self.lower[i], self.upper[i]] for i in range(self.npar) ]))
        self.show_plot = show_plot
        
        # Results directory
        self.results_dir = results_dir
        
        # Training loss, validation loss
        self.training_loss = np.array([])
        self.validation_loss = np.array([])
        self.sequential_training_loss = []
        self.sequential_validation_loss = []
        self.sequential_nsims = []

        # MPI-specific setup
        self.rank = rank
        self.n_procs = n_procs
        if n_procs > 1:
            self.use_mpi = True
            self.comm = comm
            self.red_op = red_op
        else:
            self.use_mpi = False

        # Are we in a jupyter notebook or not?
        self.nb = isnotebook()

        # Show progress bars?
        self.progress_bar = progress_bar
    
    # Divide list of jobs between MPI processes
    def allocate_jobs(self, n_jobs):
        n_j_allocated = 0
        for i in range(self.n_procs):
            n_j_remain = n_jobs - n_j_allocated
            n_p_remain = self.n_procs - i
            n_j_to_allocate = int(n_j_remain / n_p_remain)
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
            return self.log_likelihood(x) + np.log(self.prior.pdf(x))
    
    # Log posterior
    def log_geometric_mean_proposal(self, x):
        
        if self.prior.pdf(x) == 0:
            return -1e300
        else:
            return 0.5 * (self.log_likelihood(x) + 2 * np.log(self.prior.pdf(x)) )
    
    # Run n_batch simulations
    def run_simulation_batch(self, n_batch, ps, simulator, compressor, simulator_args, compressor_args, seed_generator = None, sub_batch = 1):
        
        # Random seed generator: set to unsigned 32 bit int random numbers as default
        if seed_generator is None:
            seed_generator = lambda: np.random.randint(2147483647)
    
        # Dimension outputs
        data_samples = np.zeros((n_batch*sub_batch, self.npar))
        parameter_samples = np.zeros((n_batch*sub_batch, self.npar))
        
        # Run samples assigned to each process, catching exceptions
        # (when simulator returns np.nan).
        i_prop = self.inds_prop[0]
        i_acpt = self.inds_acpt[0]
        err_msg = 'Simulator returns {:s} for parameter values: {} (rank {:d})'
        if self.progress_bar:
            if self.nb:
                pbar = tqdm.tqdm_notebook(total = self.inds_acpt[-1], desc = "Simulations")
            else:
                pbar = tqdm.tqdm(total = self.inds_acpt[-1], desc = "Simulations")
        while i_acpt <= self.inds_acpt[-1]:
            try:
                sims = simulator(ps[i_prop,:], seed_generator(), simulator_args, sub_batch)
                
                # Make sure the sims are the right shape
                if sub_batch == 1 and len(sims) != 1:
                    sims = np.array([sims])
                compressed_sims = np.array([compressor(sims[k], compressor_args) for k in range(sub_batch)])
                if np.all(np.isfinite(compressed_sims.flatten())):
                    data_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = compressed_sims
                    parameter_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = ps[i_prop,:]
                    i_acpt += 1
                    if self.progress_bar:
                        pbar.update(1)
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
    
        lnL = self.nde.eval((np.atleast_2d((theta-self.theta_fiducial)/self.fisher_errors), np.atleast_2d((self.data-self.theta_fiducial)/self.fisher_errors)), self.sess)
    
        if np.isnan(lnL) == True:
            return -1e300
        else:
            return lnL

    def sequential_training(self, simulator, compressor, n_initial, n_batch, n_populations, proposal = None, \
                            simulator_args = None, compressor_args = None, safety = 5, plot = True, batch_size = 100, \
                            validation_split = 0.1, epochs = 300, patience = 20, seed_generator = None, \
                            save_intermediate_posteriors = True, sub_batch = 1):

        # Set up the initial parameter proposal density
        if proposal is None:
            proposal = priors.TruncatedGaussian(self.data, 9*self.Finv, self.lower, self.upper)

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
        xs_batch, ps_batch = self.run_simulation_batch(n_initial, ps, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

        # Train on master only
        if self.rank == 0:

            # Construct the initial training-set
            
            ps_batch = (ps_batch - self.theta_fiducial)/self.fisher_errors
            xs_batch = (xs_batch - self.theta_fiducial)/self.fisher_errors
            self.ps = np.concatenate([self.ps, ps_batch])
            self.xs = np.concatenate([self.xs, xs_batch])
            self.x_train = self.ps.astype(np.float32)
            self.y_train = self.xs.astype(np.float32)
            self.n_sims = len(self.x_train)

            # Train the network on these initial simulations
            val_loss, train_loss = self.trainer.train(self.sess, [self.x_train, self.y_train], validation_split = validation_split, epochs = epochs, batch_size=max(self.n_sims//8, batch_size), progress_bar=self.progress_bar,
                  patience=patience, saver_name='{}tmp_model'.format(self.results_dir))
                  
            # Save the training and validation losses
            self.training_loss = np.concatenate([self.training_loss, train_loss])
            self.validation_loss = np.concatenate([self.validation_loss, val_loss])
            self.sequential_training_loss.append(train_loss[-1])
            self.sequential_validation_loss.append(val_loss[-1])
            self.sequential_nsims.append(self.n_sims)
            
            # Generate posterior samples
            if save_intermediate_posteriors:
                print('Sampling approximate posterior...')
                self.posterior_samples = self.emcee_sample(self.log_posterior, \
                                  [self.posterior_samples[-i,:] for i in range(self.nwalkers)], \
                                  main_chain=self.posterior_chain_length)
            
                # Save posterior samples to file
                f = open('{}posterior_samples_0.dat'.format(self.results_dir), 'w')
                np.savetxt(f, self.posterior_samples)
                f.close()
            
                print('Done.')

                # If plot == True, plot the current posterior estimate
                if plot == True:
                    self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='{}seq_train_post_0.pdf'.format(self.results_dir))

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
            xs_batch, ps_batch = self.run_simulation_batch(n_batch, ps_batch, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

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
                val_loss, train_loss = self.trainer.train(self.sess, [self.x_train, self.y_train], validation_split=validation_split, epochs=epochs, batch_size=max(self.n_sims//8, batch_size), progress_bar=self.progress_bar,
                           patience=patience, saver_name='{}tmp_model'.format(self.results_dir))
                           
                # Save the training and validation losses
                self.training_loss = np.concatenate([self.training_loss, train_loss])
                self.validation_loss = np.concatenate([self.validation_loss, val_loss])
                self.sequential_training_loss.append(train_loss[-1])
                self.sequential_validation_loss.append(val_loss[-1])
                self.sequential_nsims.append(self.n_sims)

                # Generate posterior samples
                if save_intermediate_posteriors:
                    print('Sampling approximate posterior...')
                    self.posterior_samples = self.emcee_sample(self.log_posterior, \
                                      [self.posterior_samples[j] for j in range(self.nwalkers)], \
                                      main_chain=self.posterior_chain_length)
                
                    # Save posterior samples to file
                    f = open('{}posterior_samples_{:d}.dat'.format(self.results_dir, i+1), 'w')
                    np.savetxt(f, self.posterior_samples)
                    f.close()

                    print('Done.')

                    # If plot == True
                    if plot == True:
                        # Plot the posterior
                        self.triangle_plot([self.posterior_samples], \
                                        savefig=True, \
                                        filename='{}seq_train_post_{:d}.pdf'.format(self.results_dir, i + 1))

                # Plot training convergence
                if plot == True:
                    # Plot the training loss convergence
                    self.sequential_training_plot(savefig=True, filename='{}seq_train_loss.pdf'.format(self.results_dir))

    def train(self, plot=True, batch_size=100, validation_split=0.1, epochs=500, patience=20):
        
        # Train the network on these initial simulations
        val_loss, train_loss = self.trainer.train(self.sess, [self.x_train, self.y_train], validation_split = validation_split, epochs=epochs, batch_size=batch_size, progress_bar=self.progress_bar, patience=patience, saver_name='{}tmp_model'.format(self.results_dir))
        
        # Save the training and validation losses
        self.training_loss = np.concatenate([self.training_loss, train_loss])
        self.validation_loss = np.concatenate([self.validation_loss, val_loss])

        # Generate posterior samples
        print('Sampling approximate posterior...')
        self.posterior_samples = \
                self.emcee_sample(self.log_posterior, \
                                  [self.posterior_samples[-i,:] for i in range(self.nwalkers)], \
                                  main_chain=self.posterior_chain_length)
        
        # Save posterior samples to file
        f = open('{}posterior_samples.dat'.format(self.results_dir), 'w')
        np.savetxt(f, self.posterior_samples)
        f.close()
        
        print('Done.')
        
        # If plot == True, plot the current posterior estimate
        if plot == True:
            self.triangle_plot([self.posterior_samples], \
                                savefig=True, \
                                filename='{}post_trained.pdf'.format(self.results_dir))

    def load_simulations(self, xs_batch, ps_batch):
        
        ps_batch = (ps_batch - self.theta_fiducial)/self.fisher_errors
        xs_batch = (xs_batch - self.theta_fiducial)/self.fisher_errors
        self.ps = np.concatenate([self.ps, ps_batch])
        self.xs = np.concatenate([self.xs, xs_batch])
        self.x_train = self.ps.astype(np.float32)
        self.y_train = self.xs.astype(np.float32)
        self.n_sims += len(ps_batch)
    
    def fisher_pretraining(self, n_batch=50000, plot=True, batch_size=100, validation_split=0.1, epochs=300, patience=10):

        # Train on master only
        if self.rank == 0:

            # Generate fisher pre-training data
            
            # Broader proposal
            proposal = priors.TruncatedGaussian(self.theta_fiducial, 9*self.Finv, self.lower, self.upper)

            # Anticipated covariance of the re-scaled data
            Cdd = np.zeros((self.npar, self.npar))
            for i in range(self.npar):
                for j in range(self.npar):
                    Cdd[i,j] = self.Finv[i,j]/(self.fisher_errors[i]*self.fisher_errors[j])
            Ldd = np.linalg.cholesky(Cdd)
            Cddinv = np.linalg.inv(Cdd)
            ln2pidetCdd = np.log(2*np.pi*np.linalg.det(Cdd))
            
            # Sample parameters from some broad proposal
            ps = np.zeros((3*n_batch, self.npar))
            for i in range(0, n_batch):
                # Draws from prior
                ps[i,:] = (self.prior.draw() - self.theta_fiducial)/self.fisher_errors
                
                # Draws from asymptotic posterior
                ps[n_batch + i,:] = (self.asymptotic_posterior.draw() - self.theta_fiducial)/self.fisher_errors
                
                # Drawn from Gaussian with 3x anticipated covariance matrix
                ps[2*n_batch + i,:] = (proposal.draw() - self.theta_fiducial)/self.fisher_errors
            
            # Sample data assuming a Gaussian likelihood
            xs = np.array([pss + np.dot(Ldd, np.random.normal(0, 1, self.npar)) for pss in ps])

            # Construct the initial training-set
            fisher_x_train = ps.astype(np.float32).reshape((3*n_batch, self.npar))
            fisher_y_train = xs.astype(np.float32).reshape((3*n_batch, self.npar))
            
            # Train network on initial (asymptotic) simulations
            
            # Train the network on these initial simulations
            validation_loss, train_loss = self.trainer.train(self.sess, [fisher_x_train, fisher_y_train], validation_split = validation_split, epochs=epochs, batch_size=batch_size, progress_bar=self.progress_bar, patience=patience, saver_name='{}tmp_model'.format(self.results_dir))
            
            # Initialization for the EMCEE sampling
            x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]

            print('Sampling approximate posterior...')
            self.posterior_samples = self.emcee_sample(self.log_posterior, x0, main_chain=self.posterior_chain_length)
            print('Done.')

            # if plot == True
            if plot == True:
                self.triangle_plot([self.posterior_samples], \
                                    savefig=True, \
                                    filename='{}fisher_pretrain_post.pdf'.format(self.results_dir))

    def triangle_plot(self, samples, savefig = False, filename = None):

        mc_samples = [MCSamples(samples=s, names = self.names, labels = self.labels, ranges = self.ranges) for s in samples]

        # Triangle plot
        g = plots.getSubplotPlotter(width_inch = 12)
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.6
        g.settings.axes_fontsize=14
        g.settings.legend_fontsize=16
        g.settings.lab_fontsize=20
        g.triangle_plot(mc_samples, filled_compare=True, normalized=True)
        for i in range(0, len(samples[0][0,:])):
            for j in range(0, i+1):
                ax = g.subplots[i,j]
                xtl = ax.get_xticklabels()
                ax.set_xticklabels(xtl, rotation=45)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        
        if savefig:
            plt.savefig(filename)
        if self.show_plot:
            plt.show()
        else:
            plt.close()

    def sequential_training_plot(self, savefig = False, filename = None):

        plt.close()
        columnwidth = 18 # cm
        aspect = 1.67
        pts_per_inch = 72.27
        inch_per_cm = 2.54
        width = columnwidth/inch_per_cm
        plt.rcParams.update({'figure.figsize': [width, width / aspect],
                          'backend': 'pdf',
                          'font.size': 15,
                          'legend.fontsize': 15,
                          'legend.frameon': False,
                          'legend.loc': 'best',
                          'lines.markersize': 3,
                          'lines.linewidth': .5,
                          'axes.linewidth': .5,
                          'axes.edgecolor': 'black'})
                          
        # Trace plot of the training and validation loss as a function of the number of simulations ran
        plt.scatter(self.sequential_nsims, self.sequential_training_loss, s = 20, alpha = 0.5, color = 'red')
        plt.plot(self.sequential_nsims, self.sequential_training_loss, color = 'red', lw = 2, alpha = 0.5, label = 'training loss')
        plt.scatter(self.sequential_nsims, self.sequential_validation_loss, s = 20, alpha = 0.5, color = 'blue')
        plt.plot(self.sequential_nsims, self.sequential_validation_loss, color = 'blue', lw = 2, alpha = 0.5, label = 'validation loss')

        plt.xlabel(r'number of simulations, $n_\mathrm{sims}$')
        plt.ylabel(r'negative log loss, $-\mathrm{ln}\,U$')
        plt.tight_layout()
        plt.legend()

        if savefig:
            plt.savefig(filename)
        if self.show_plot:
            plt.show()
        else:
            plt.close()
