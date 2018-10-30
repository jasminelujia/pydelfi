from scipy.stats import multivariate_normal
import numpy as np

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

# Divide list of jobs between MPI processes
def allocate_jobs(rank, n_procs, n_jobs):
    n_j_allocated = 0
    for i in range(n_procs):
        n_j_remain = n_jobs - n_j_allocated
        n_p_remain = n_procs - i
        n_j_to_allocate = int(n_j_remain / n_p_remain)
        if rank == i:
            return range(n_j_allocated, \
                         n_j_allocated + n_j_to_allocate)
        n_j_allocated += n_j_to_allocate

# Combine arrays from all processes assuming
# 1) array was initially zero
# 2) each process has edited a unique slice of the array
def complete_array(comm, target_distrib):
    target = np.zeros(target_distrib.shape, dtype=target_distrib.dtype)
    comm.Allreduce(target_distrib, target, op=red_op)

    return target

def mean_covariance(simulator, theta_fiducial, nsims, ndata, simulator_args = None, seed_generator = None, use_mpi = True):

    # Set the random seed generator
    if seed_generator is not None:
        self.seed_generator = seed_generator
    else:
        self.seed_generator = lambda: np.random.randint(2147483647)

    sims = np.zeros((nsims, ndata))

    # Allocate jobs according to MPI
    inds = allocate_jobs(rank, nsims, n_jobs)

    # Run the simulations
    for i in range(inds[-1]):
        seed = seed_generator()
        sims[i,:] = simulation(theta_fiducial, seed, sim_args)

    # Collect all the sims together from all the processes
    sims = complete_array(comm, sims)

    # Now compute the covariance and mean
    C = np.zeros((ndata,ndata))
    mu = np.zeros((ndata))
    mu2 = np.zeros((ndata, ndata))
    for i in range(0, nsims):
        mu += sims[i,:]/nsims
        mu2 += np.outer(sims[i,:], sims[i,:])/nsims
    C = mu2 - np.outer(mu,mu)
    
    return mu, C, sims


def mean_derivatives(simulator, theta_fiducial, h, nsims = 1, simulator_args = None, seed_generator = None):

    # Set the random seed generator
    if seed_generator is not None:
        self.seed_generator = seed_generator
    else:
        self.seed_generator = lambda: np.random.randint(2147483647)

    # Initialize dmudt
    npar = len(theta_fiducial)
    dmudt = [0]*npar
    
    # Run seed matched simulations for derivatives
    for k in range(nsims):
        
        # Set random seed
        seed = seed_generator()
        
        # Fiducial simulation
        d_fiducial = self.simulation(theta_fiducial, seed, sim_args)
                
        # Loop over parameters
        for i in range(0, npar):
                
            # Step theta
            theta = np.copy(theta_fiducial)
            theta[i] += h[i]
                
            # Shifted simulation with same seed
            d_dash = self.simulation(theta, seed, simulation_args)
                
            # Forward step derivative
            dmudt[i] += ( (d_dash - d_fiducial)/h[i] )/nsims

    return np.array(dmudt)


class Gaussian():

    def __init__(self, ndata, theta_fiducial, mu = None, Cinv = None, dmudt = None, dCdt = None, F = None, prior_mean = None, prior_covariance = None, rank=0, n_procs=1, comm=None, red_op=None):
    
        # Load inputs
        self.theta_fiducial = theta_fiducial
        self.npar = len(theta_fiducial)
        self.ndata = len(mu)
        self.mu = mu
        self.Cinv = Cinv
        self.dmudt = dmudt
        self.dCdt = dCdt
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        if F is None:
            self.F = None
            self.Finv = None
        else:
            self.F = F
            self.Finv = np.linalg.inv(F)
        
        # Holder to store any simulations and parameter values that get ran
        self.simulations = np.array([]).reshape((0,self.ndata))
        self.parameters = np.array([]).reshape((0,self.npar))
        
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
    
    # Compute the mean and covariance
    def compute_mean_covariance(self, simulator, nsims, simulator_args = None, seed_generator = None, progress_bar=True):
    
        # Set the random seed generator
        if seed_generator is not None:
            seed_generator = seed_generator
        else:
            seed_generator = lambda: np.random.randint(2147483647)

        sims = np.zeros((self.nsims, self.ndata))

        # Allocate jobs according to MPI
        inds = allocate_jobs(nsims)

        # Run the simulations with MPI
        if progress_bar:
            if self.nb:
                pbar = tqdm.tqdm_notebook(total = inds[-1], desc = "Covariance simulations")
            else:
                pbar = tqdm.tqdm(total = inds[-1], desc = "Covariance simulations")
        for i in range(inds[-1]):
            seed = seed_generator()
            sims[i,:] = simulator(self.theta_fiducial, seed, simulator_args)
            if progress_bar:
                pbar.update(1)

        # Collect all the sims together from all the processes
        sims = self.complete_array(sims)

        # Now compute the covariance and mean
        self.mu = np.zeros((self.ndata))
        mu2 = np.zeros((self.ndata, self.ndata))
        for i in range(0, nsims):
            mu += sims[i,:]/nsims
            mu2 += np.outer(sims[i,:], sims[i,:])/nsims
        self.C = mu2 - np.outer(mu,mu)

        # Save the simulations
        self.simulations = sims
        self.parameters = np.array([self.theta_fiducial for i in range(nsims)])
            
    def compute_derivatives(self, simulator, nsims, h, simulator_args = None, seed_generator = None, progress_bar=True):
    
        # Set the random seed generator
        if seed_generator is not None:
            seed_generator = seed_generator
        else:
            seed_generator = lambda: np.random.randint(2147483647)

        # Initialize the derivatives
        self.dmudt = np.zeros((self.npar, self.ndata))

        # Run seed matched simulations for derivatives
        if progress_bar:
            if self.nb:
                pbar = tqdm.tqdm_notebook(total = nsims*self.npar, desc = "Derivative simulations")
            else:
                pbar = tqdm.tqdm(total = nsims*self.npar, desc = "Derivative simulations")
        for k in range(nsims):
            
            # Set random seed
            seed = seed_generator()
            
            # Fiducial simulation
            d_fiducial = simulator(self.theta_fiducial, seed, simulator_args)
            
            # Loop over parameters
            for i in range(0, self.npar):
                
                # Step theta
                theta = np.copy(self.theta_fiducial)
                theta[i] += h[i]
                
                # Shifted simulation with same seed
                d_dash = simulator(theta, seed, simulator_args)
                self.simulations = np.concatenate([self.simulations, d_dash])
                self.parameters = np.concatenate([self.parameters, theta])

                # Forward step derivative
                self.dmudt[i,:] += ( (d_dash - d_fiducial)/h[i] )/nsims

                if progress_bar:
                    pbar.update(1)
        

    # Fisher score maximum likelihood estimator
    def scoreMLE(self, d):
    
        # Compute the score
        dLdt = np.zeros(self.npar)
    
        # Add terms from mean derivatives
        for a in range(self.npar):
            dLdt[a] += np.dot(self.dmudt[a,:], np.dot(self.Cinv, (d - self.mu)))
                
        # Add terms from covariance derivatives
        if self.dCdt is not None:
            for a in range(self.npar):
                dLdt[a] += -0.5*np.trace(np.dot(self.Cinv, self.dCdt[a,:,:])) + 0.5*np.dot((d - self.mu), np.dot( np.dot(self.Cinv, np.dot(self.dCdt[a,:,:], self.Cinv)), (d - self.mu)))

        # Cast to MLE
        t = self.theta_fiducial + np.dot(self.Finv, dLdt)
        
        # Correct for gaussian prior if one is provided
        if self.prior_mean is not None:
            t += np.dot(self.Finv, np.dot(np.linalg.inv(self.prior_covariance), self.prior_mean - self.theta_fiducial))

        return t

    # Fisher matrix
    def compute_fisher(self):
    
        # Fisher matrix
        F = np.zeros((self.npar, self.npar))
    
        # Mean derivatives part
        for a in range(0, self.npar):
            for b in range(0, self.npar):
                F[a, b] += 0.5*(np.dot(self.dmudt[a,:], np.dot(self.Cinv, self.dmudt[b,:])) + np.dot(self.dmudt[b,:], np.dot(self.Cinv, self.dmudt[a,:])))
        
        # Covariance derivatives part
        if self.dCdt is not None:
            for a in range(0, self.npar):
                for b in range(0, self.npar):
                    F[a, b] += 0.5*np.trace( np.dot( np.dot(self.Cinv, self.dCdt[a,:,:]), np.dot(self.Cinv, self.dCdt[b,:,:]) ) )

        # Add the prior covariance if one is provided
        if self.prior_covariance is not None:
            F = F + np.linalg.inv(self.prior_covariance)

        self.F = F
        self.Finv = np.linalg.inv(F)

    # Nuisance projected score
    def projected_scoreMLE(self, d, nuisances):
        
        # indices for interesting parameters
        interesting = np.delete(np.arange(self.npar), nuisances)
        n_interesting = len(interesting)
        n_nuisance = len(nuisances)
        
        # Compute projection vectors
        P = np.zeros((n_interesting, n_nuisance))
        Fnn_inv = np.linalg.inv(np.delete(np.delete(self.F, interesting, axis = 0), interesting, axis = 1))
        Finv_tt = np.delete(np.delete(self.Finv, nuisances, axis=0), nuisances, axis=1)
        for i in range(n_interesting):
            P[i,:] = np.dot(Fnn_inv, self.F[i,nuisances])

        # Compute the score
        dLdt = np.zeros(self.npar)
    
        # Add terms from mean derivatives
        for a in range(self.npar):
            dLdt[a] += np.dot(self.dmudt[a,:], np.dot(self.Cinv, (d - self.mu)))
        
        # Add terms from covariance derivatives
        if self.dCdt is not None:
            for a in range(self.npar):
                dLdt[a] += -0.5*np.trace(np.dot(self.Cinv, self.dCdt[a,:,:])) + 0.5*np.dot((d - self.mu), np.dot( np.dot(self.Cinv, np.dot(self.dCdt[a,:,:], self.Cinv)), (d - self.mu)))

        # Do the projection
        dLdt_projected = np.zeros(n_interesting)
        for a in range(n_interesting):
            dLdt_projected[a] = dLdt[a] - np.dot(P[a], dLdt[nuisances])

        # Cast it back into an MLE
        t = np.dot(Finv_tt, dLdt_projected) + self.theta_fiducial[interesting]

        # Correct for the prior if one is provided
        if self.prior_mean is not None:
            Qinv_tt = np.delete(np.delete(np.linalg.inv(self.prior_covariance), nuisances, axis=0), nuisances, axis=1)
            t += np.dot(Finv_tt, np.dot(Qinv_tt, self.prior_mean[interesting] - self.theta_fiducial[interesting]))

        return t




class Wishart():

    def __init__(self, theta_fiducial, nu, Cinv, dCdt, F = None, prior_mean = None, prior_covariance = None):
        
        # Load inputs
        self.theta_fiducial = theta_fiducial
        self.npar = len(theta_fiducial)
        self.ndata = len(Cinv)
        self.Cinv = Cinv
        self.dCdt = dCdt
        self.nu = nu
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance

        # Compute the Fisher matrix or use pre-loaded
        if F is not None:
            self.F = F
        else:
            self.F = self.fisher()
        self.Finv = np.linalg.inv(self.F)

    # Fisher score maximum likelihood estimator
    def scoreMLE(self, d):
    
        # Compute the score
        dLdt = np.zeros(self.npar)
        for a in range(self.npar):
            for l in range(self.ndata):
                dLdt[a] += self.nu[l]*(-0.5*np.trace(np.dot(self.Cinv[l,:,:], self.dCdt[a,l,:,:])) + 0.5*np.trace(np.dot( np.dot(self.Cinv[l,:,:], np.dot(self.dCdt[a,l,:,:], self.Cinv[l,:,:])), d[l,:,:]) ) )

        # Make it an MLE
        t = np.dot(self.Finv, dLdt) + self.theta_fiducial

        # Correct for prior if there is one
        if self.prior_covariance is not None:
            t += np.dot(self.Finv, np.dot(np.linalg.inv(self.prior_covariance), self.prior_mean - self.theta_fiducial))
    
        # Return summary statistics
        return t

    # Fisher matrix
    def fisher(self):
    
        # Fisher matrix
        F = np.zeros((self.npar, self.npar))
        for a in range(self.npar):
            for b in range(self.npar):
                for l in range(self.ndata):
                    F[a,b] += 0.5*self.nu[l]*np.trace( np.dot(self.Cinv[l,:,:], np.dot(self.dCdt[a,l,:,:], np.dot(self.Cinv[l,:,:], self.dCdt[b,l,:,:]) ) ))

        # Add prior covariance if there is one
        if self.prior_covariance is not None:
            F = F + np.linalg.inv(self.prior_covariance)

        return F

    # Nuisance projected score
    def projected_scoreMLE(self, d, nuisances):
        
        # indices for interesting parameters
        interesting = np.delete(np.arange(self.npar), nuisances)
        n_interesting = len(interesting)
        n_nuisance = len(nuisances)
        
        # Compute projection vectors
        P = np.zeros((n_interesting, n_nuisance))
        Fnn_inv = np.linalg.inv(np.delete(np.delete(self.F, interesting, axis = 0), interesting, axis = 1))
        Finv_tt = np.delete(np.delete(self.Finv, nuisances, axis=0), nuisances, axis=1)
        for i in range(n_interesting):
            P[i,:] = np.dot(Fnn_inv, self.F[i,nuisances])

        # Compute the score
        dLdt = np.zeros(self.npar)
        for a in range(self.npar):
            for l in range(self.ndata):
                dLdt[a] += self.nu[l]*(-0.5*np.trace(np.dot(self.Cinv[l,:,:], self.dCdt[a,l,:,:])) + 0.5*np.trace(np.dot( np.dot(self.Cinv[l,:,:], np.dot(self.dCdt[a,l,:,:], self.Cinv[l,:,:])), d[l,:,:]) ) )

        # Do the projection
        dLdt_projected = np.zeros(n_interesting)
        for a in range(n_interesting):
            dLdt_projected[a] = dLdt[a] - np.dot(P[a], dLdt[nuisances])

        # Cast it back into an MLE
        t = np.dot(Finv_tt, dLdt_projected) + self.theta_fiducial[interesting]

        # Correct for the prior if one is provided
        if self.prior_mean is not None:
            Qinv_tt = np.delete(np.delete(np.linalg.inv(self.prior_covariance), nuisances, axis=0), nuisances, axis=1)
            t += np.dot(Finv_tt, np.dot(Qinv_tt, self.prior_mean[interesting] - self.theta_fiducial[interesting]))

        return t

