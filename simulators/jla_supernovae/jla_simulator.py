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

def compressor(data, args):

    theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean = args
    
    return mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, data)

def compressor_projected(data, args):
    
    theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, F, P1, P2 = args

    # MOPED compress the data
    d_twidle = mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, data)
    
    # Now do the projection
    d_twidle = np.dot(F, d_twidle - theta_fiducial - np.dot(Finv, np.dot(Qinv, prior_mean - theta_fiducial)))
    d_twidle = np.dot(Finv[0:2, 0:2], np.array([d_twidle[0] - np.dot(P1, d_twidle[2:]), d_twidle[1] - np.dot(P2, d_twidle[2:])]))
    d_twidle = d_twidle + theta_fiducial[:2] + np.dot(Finv[:2,:2], np.dot(Qinv[:2,:2], prior_mean[:2] - theta_fiducial[:2]))

    return d_twidle

class JLA():

    def __init__():

        # Import data
        self.jla_data, self.jla_cmats = jla_parser.b14_parse(z_min=None, z_max=None, qual_cut=False,
                                           jla_path='simulators/jla_supernovae/jla_data/')
        self.data = jla_data['mb']
        delta_m_cut = 10
        self.auxiliary_data = np.column_stack([jla_data['zcmb'], jla_data['x1'], jla_data['color'], np.array([(jla_data['3rdvar'] > delta_m_cut)], dtype=int)[0]])

        # Om, w0, M_b, alpha, beta, delta_m
        self.npar = 6
        self.theta_fiducial = np.array([  0.20181324,  -0.74762939, -19.04253368,   0.12566322,   2.64387045, -0.05252869])

        # Covariance matrix
        self.C = jla_parser.b14_covariance(jla_data, jla_cmats, theta_fiducial[3], theta_fiducial[4])
        self.Cinv = np.linalg.inv(C)
        self.L = np.linalg.cholesky(C)

        # Derivative of the covariance matrix
        self.n_sn = len(C)
        self.dCdt = np.zeros((self.npar, self.n_sn, self.n_sn))

        # Step size for derivatives
        self.step = abs(0.01*self.theta_fiducial)

        # N data points
        nself.data = len(self.jla_data['mb'])

        # Simulation args
        sim_args = [auxiliary_data, L]

        # Compute the mean
        mu = jla.apparent_magnitude(theta_fiducial, auxiliary_data)

        # Compute the derivatives
        dmdt = jla.dmudtheta(theta_fiducial, jla.simulation_seeded, step, npar, ndata, sim_args)
        dmdt[2,:] = np.ones(n_sn)
        dmdt[3,:] = -jla_data['x1']
        dmdt[4,:] = jla_data['color']
        dmdt[5,:] = (jla_data['3rdvar'] > 10)

        # Fisher matrix
        F, Finv = jla.fisher(dmdt, dCdt, Cinv, Qinv, npar)
        fisher_errors = np.sqrt(np.diag(Finv))





