import numpy as np
import scipy.integrate as integrate
from .moped import *
from scipy.stats import multivariate_normal

def flat_prior(theta, prior_args):
    
    lower = prior_args[0]
    upper = prior_args[1]
    
    # Test each parameter against prior limits
    for a in range(0, len(theta)):
        
        if theta[a] > upper[a] or theta[a] < lower[a]:
            return 0
        
    return np.prod(upper-lower)

def gaussian_prior_draw(prior_args):
    
    mean = prior_args[0]
    C = prior_args[1]
    return multivariate_normal.rvs(mean=mean, cov=C)


def truncated_gaussian_prior_draw(prior_args):
    
    mean = prior_args[0]
    C = prior_args[1]
    lower = prior_args[2]
    upper = prior_args[3]
    
    # While outside possible range
    P = 0
    while P == 0:
        x = gaussian_prior_draw([mean, C])
        P = flat_prior(x, [lower, upper])
    return x

# Distance modulus
def apparent_magnitude(theta, auxiliary_data):
    
    # Cosmological parameters
    Om = theta[0]
    w0 = theta[1]
    h = 0.7
    
    # Systematics parameters
    Mb = theta[2]
    alpha = theta[3]
    beta = theta[4]
    delta_m = theta[5]
    
    # Pull out the relevant things from the data
    z = auxiliary_data[:,0]
    x = auxiliary_data[:,1]
    c = auxiliary_data[:,2]
    v3 = auxiliary_data[:,3]
    
    # Holders
    distance_modulus = np.zeros(len(z))
    
    for i in range(len(z)):
        integral = integrate.quad(lambda zz: 1./np.sqrt( Om*(1+zz)**3 + (1-Om)*(1+zz)**(3*(1+w0)) ), 0, z[i])[0]
        distance_modulus[i] = 25 - 5*np.log10(h) + 5*np.log10(3000*(1+z[i])*integral)
    
    return Mb - alpha*x + beta*c + delta_m*v3 + distance_modulus

# Generate realisation of \mu
def simulation(theta, sim_args):
    
    # Pull out data
    auxiliary_data = sim_args[0]
    L = sim_args[1]
    
    # Signal
    mb = apparent_magnitude(theta, auxiliary_data)
        
    # Noise
    noise = np.dot(L, np.random.normal(0, 1, len(L)))
    
    # Return signal + noise
    return mb + noise

# Generate realisation of \mu
def simulation_seeded(theta, seed, sim_args):
    
    # Pull out data
    auxiliary_data = sim_args[0]
    L = sim_args[1]
    
    # Signal
    mb = apparent_magnitude(theta, auxiliary_data)
        
    # Noise
    np.random.seed(seed)
    noise = np.dot(L, np.random.normal(0, 1, len(L)))
    
    # Return signal + noise
    return mb

def simulationABC(theta, simABC_args):
    
    # Extract args
    theta_fiducial = simABC_args[0]
    Finv = simABC_args[1]
    Cinv = simABC_args[2]
    dmdt = simABC_args[3]
    dCdt = simABC_args[4]
    mu = simABC_args[5]
    Sinv = simABC_args[6]
    mu_prior = simABC_args[7]
    sim_args = simABC_args[8]
    P1 = simABC_args[9]
    P2 = simABC_args[10]
    F = simABC_args[11]
    prior_args = simABC_args[12]
    
    # Draw nuisance parameters
    nuisances = truncated_gaussian_prior_draw(prior_args)[2:]
    
    # Simulate data
    d = simulation(np.concatenate([theta, nuisances]), sim_args)
    
    # MOPED compress the data
    d_twidle = mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Sinv, mu_prior, d)
    
    # Now do the projection
    d_twidle = np.dot(F, d_twidle - theta_fiducial - np.dot(Finv, np.dot(Sinv, mu_prior - theta_fiducial)))
    d_twidle = np.dot(Finv[0:2, 0:2], np.array([d_twidle[0] - np.dot(P1, d_twidle[2:]), d_twidle[1] - np.dot(P2, d_twidle[2:])]))
    d_twidle = d_twidle + theta_fiducial[:2] + np.dot(Finv[:2,:2], np.dot(Sinv[:2,:2], mu_prior[:2] - theta_fiducial[:2]))
    
    return d_twidle
