import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from simulators.cosmic_shear.cosmic_shear_map import *
import ndes.nde as nde
import distributions.priors as priors

import numpy.random as npr
import matplotlib.cm as mpcm
import healpy as hp

# settings
use_mpi = True

# set up MPI environment if desired
if use_mpi:
    import mpi4py.MPI as mpi
    comm = mpi.COMM_WORLD
    n_procs = comm.Get_size()
    rank = comm.Get_rank()
    red_op = mpi.SUM
else:
    n_procs = 1
    rank = 0
    comm = None
    red_op = None


# Create the DELFI MDN object
n_components = 1

mdn = nde.DelfiMixtureDensityNetwork(simulator, prior, \
                                     asymptotic_posterior, f_mat_inv, \
                                     theta_fiducial, data, \
                                     n_components, simulator_args = sim_args, \
                                     n_hidden = [50, 50], \
                                     activations = ['tanh', 'tanh'], \
                                     names = names, labels = labels, \
                                     ranges = ranges, rank = rank, \
                                     n_procs = n_procs, comm = comm, \
                                     red_op = red_op, show_plot = False)

# Do the Fisher pre-training
#mdn.fisher_pretraining(50000, prior, epochs=50)
#mdn.fisher_pretraining(50000, prior, epochs=25)
mdn.fisher_pretraining(50000, prior, epochs=10)

# Proposal for the SNL
proposal = priors.TruncatedGaussian(theta_fiducial, 9*f_mat_inv, lower, upper)

'''
# Tests of MPI capability and error checking
print 'random check', rank, npr.randn(5)
props = -np.ones((10, len(theta_fiducial)))
props[1, :] = theta_fiducial[:]
props[3, :] = theta_fiducial[:]
props[5, :] = theta_fiducial[:]*1.01
props[7, :] = theta_fiducial[:]
print rank, props
mdn.inds_prop = mdn.allocate_jobs(5 * 2)
mdn.inds_acpt = mdn.allocate_jobs(2)
omg, zomg = mdn.run_simulation_batch(2, props)
print rank, 'data samples:', omg
print rank, 'param samples:', zomg
exit()
'''

# Initial samples, batch size for population samples, number of populations
n_initial = 50#500
n_batch = 50#500
n_populations = 8

# Do the SNL training
mdn.sequential_training(n_initial, n_batch, n_populations, proposal)

# Trace plot of the loss as a function of the number of simulations
if rank == 0:
    plt.scatter(mdn.n_sim_trace, mdn.loss_trace, s = 20)
    plt.plot(mdn.n_sim_trace, mdn.loss_trace, color = 'red')
    plt.xlim(0, mdn.n_sim_trace[-1])
    plt.xlabel('number of simulations')
    plt.ylabel('loss')
    plt.show()

