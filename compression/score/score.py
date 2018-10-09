from scipy.stats import multivariate_normal
import numpy as np

def mean_covariance(simulator, theta_fiducial, nsims, simulator_args = None, seed_generator = None):

    # Set the random seed generator
    if seed_generator is not None:
        self.seed_generator = seed_generator
    else:
        self.seed_generator = lambda: np.random.randint(2147483647)

    C = np.zeros((ndata,ndata))
    sims = np.zeros((nsims, ndata))
    mu = np.zeros((ndata))
    mu2 = np.zeros((ndata, ndata))
    for i in range(0, nsims):
        seed = seed_generator()
        sim = simulation(theta_fiducial, seed, sim_args)
        mu += sim/nsims
        mu2 += np.outer(sim, sim)/nsims
        sims[i,:] = sim
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
            d_dash = self.simulation(theta, seed, self.simulation_args)
                
            # Forward step derivative
            dmudt[i] += ( (d_dash - d_fiducial)/h[i] )/nsims

    return np.array(dmudt)


class Gaussian():

    def __init__(self, theta_fiducial, mu, Cinv, dmudt, dCdt = None, F = None, prior_mean = None, prior_covariance = None):
    
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
    
        # Add terms from mean derivatives
        for a in range(self.npar):
            dLdt[a] += np.dot(self.dmdt[a,:], np.dot(self.Cinv, (d - self.mu)))
                
        # Add terms from covariance derivatives
        if self.dCdt is not None:
            for a in range(npar):
                dLdt[a] += -0.5*np.trace(np.dot(self.Cinv, self.dCdt[a,:,:])) + 0.5*np.dot((data - self.mu), np.dot( np.dot(self.Cinv, np.dot(self.dCdt[a,:,:], self.Cinv)), (data - self.mu)))

        # Cast to MLE
        t = self.theta_fiducial + np.dot(self.Finv, dLdt)
        
        # Correct for gaussian prior if one is provided
        if self.prior_mean is not None:
            t += np.dot(self.Finv, np.dot(np.linalg.inv(self.prior_covariance), self.prior_mean - self.theta_fiducial))

        return t

    # Fisher matrix
    def fisher(self):
    
        # Fisher matrix
        F = np.zeros((self.npar, self.npar))
    
        # Mean derivatives part
        for a in range(0, self.npar):
            for b in range(0, self.npar):
                F[a, b] += 0.5*(np.dot(self.dmdt[a,:], np.dot(self.Cinv, self.dmdt[b,:])) + np.dot(self.dmdt[b,:], np.dot(self.Cinv, self.dmdt[a,:])))
        
        # Covariance derivatives part
        if self.dCdt is not None:
            for a in range(0, self.npar):
                for b in range(0, self.npar):
                    F[a, b] += 0.5*np.trace( np.dot( np.dot(self.Cinv, self.dCdt[a,:,:]), np.dot(self.Cinv, self.dCdt[b,:,:]) ) )

        # Add the prior covariance if one is provided
        if self.prior_covariance is not None:
            F = F + np.linalg.inv(self.prior_covariance)

        return F


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

