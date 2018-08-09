import numpy as np
import scipy.integrate as integrate
from .moped import *

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
    
    # Simulate data
    d = simulation(theta, sim_args)
    
    # MOPED compress the data
    d_twidle = mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Sinv, mu_prior, d)
    
    return d_twidle
