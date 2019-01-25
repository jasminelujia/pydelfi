# Delfi with iPyParallel

```python
%load_ext autoreload
%autoreload 2
```
First we load our iPyParallel engines and create a load balanced view of each engine

```python
import ipyparallel as ipp
rc = ipp.Client()
view = rc[:]
```

We will `cd` into the current directory and load all the local modules

```python
%cd /hdd/delfi
%matplotlib inline
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import distributions.priors as priors
import ndes.ndes as ndes
import delfi.delfi as delfi
import distributions.priors as priors
import compression.score.score as score
import tqdm
```
We also want to `cd` each of the engines into the current directory
```python
%%px
%cd /hdd/delfi
```
and finally we will load the modules which are needed both locally and on the engines
```python
%%px --local
%env OMP_NUM_THREADS=1
%env omp_num_threads=1
%env MKL_NUM_THREADS=1
%env mkl_num_threads=1
import numpy as np
import pickle
import healpy as hp
from simulators.cosmic_shear_map.cosmic_shear import simulate
np.seterr(invalid='raise')
```

## Define the simulator
Let's set up the simulator for cosmic shear. We first define fiducial parameter values for cosmology ($\Omega_m$, $S_8$, $\Omega_b$, $h$ and $n_s$) locally and the engine
```python
%%px --local
theta_fiducial = np.array([0.3, 0.8, 0.05, 0.70, 0.96])
npar = theta_fiducial.shape[0]
```
and we set the prior as a truncated Gaussian with upper and lower bounds, mean and covariance of
```python
lower = np.array([0, 0.4, 0, 0.4, 0.7])
upper = np.array([1, 1.2, 0.1, 1.0, 1.3])

prior_mean = np.array([0.3, 0.8, 0.05, 0.70, 0.96])
prior_covariance = np.eye(5)*np.array([0.1, 0.1, 0.05, 0.3, 0.3])**2

prior = priors.TruncatedGaussian(prior_mean, prior_covariance, lower, upper)
```
We'll also set some names and ranges for plotting
```python
names = ['\Omega_m', 'S_8', '\Omega_b', 'h', 'n_s']
labels =  ['\\Omega_m', 'S_8', '\\Omega_b', 'h', 'n_s']
ranges = {'\Omega_m':[lower[0], upper[0]],
          'S_8':[lower[1],upper[1]],
          '\Omega_b':[lower[2],upper[2]],
          'h':[lower[3],upper[3]],
          'n_s':[lower[4],upper[4]]}
```
Now we load the redshift distributions on the engines and locally

```python
%%px --local
pz_fid = pickle.load(open('simulators/cosmic_shear_map/pz_5bin.pkl', 'rb'))
nz = len(pz_fid)
```

Now we'll set the resolution of the simulation. <div style="color: red">We're currently using `nside = 32` for speed.</div>
```python
%%px --local
nside = 32
lmax = 3*nside-1
lmin = 10
n_ell_bins = 7
npix = hp.nside2npix(nside)
```

We also apply the Euclid mask
```python
%%px --local
mask = hp.ud_grade(
  hp.read_map(
    'simulators/cosmic_shear_map/Euclid_footprint_and_StarMask512.fits')
, nside)
mask = np.array([mask for x in range(nz)])
```
```
> NSIDE = 512
> ORDERING = RING in fits file
> INDXSCHM = IMPLICIT
```

We define the pixel-space shape noise standard deviation at each redshift
```python
%%px --local
sigma_e = 0.3
n_p_mean = 1.6e9 / np.sum(mask)
sig_n_p = sigma_e**2/np.random.poisson(n_p_mean, size=(nz, npix))
```

The simulator is then defined as
```python
%%px --local
def simulator(theta, seed, simulator_args, batch):
    return simulate(theta, seed, simulator_args, batch)
```
with arguments
```python
%%px --local
simulator_args = [pz_fid, lmin, lmax, sig_n_p, mask, n_ell_bins]
```

For convenience we will define a seed generator on every engine and locally
```python
%%px --local
seed_generator = lambda: np.random.randint(2147483647)
```
### Generate the observed data
Now lets perform a simulation of the _observered_ data and bench mark it.
```python
start = time.time()
data = simulator(theta_fiducial, 0, simulator_args, 1)[0]
print("time taken for a single simulation is " + str(time.time()-start) + "s")
```
```
> Sigma is 0.000000 arcmin (0.000000 rad)
> -> fwhm is 0.000000 arcmin
> Sigma is 0.000000 arcmin (0.000000 rad)
> -> fwhm is 0.000000 arcmin
> Sigma is 0.000000 arcmin (0.000000 rad)
> -> fwhm is 0.000000 arcmin
> Sigma is 0.000000 arcmin (0.000000 rad)
> -> fwhm is 0.000000 arcmin
> Sigma is 0.000000 arcmin (0.000000 rad)
> -> fwhm is 0.000000 arcmin
> time taken for a single simulation is 0.9709579944610596s
```
## Calculate score compression
### Define external functions on engines
We can parallelise the simulations (and calculation of the derivative of the mean) by running the functions independently on the engines. To do this we define the functions (which have almost the same form as in compression/score/score.py - perhaps they could be redefined this way). We have one function for the derivative of the means and one for the means and covariances at the fiducial parameter. We define the functions themselves on the engines.
```python
%%px --local
def sub_batch_sims(theta):
    seed = seed_generator()
    try:
        sims = simulator(theta, seed, simulator_args, sub_batch)
    except:
        return None
    if sub_batch == 1:
        return np.squeeze(sims, axis = 0), theta
    else:
        return sims, np.array([theta for k in range(sub_batch)])
```
```python
%%px --local
def derivative_sub_batch_sims(sim_id):
    sims_dash = np.zeros((sub_batch, ndata))
    theta = np.zeros((sub_batch, npar))
    dmudt = np.zeros((npar, ndata))
    seed = seed_generator()
    d_fiducial = np.mean(
      np.atleast_2d(
        simulator(theta_fiducial, seed, simulator_args, sub_batch)), axis=0)
    for i in range(npar):
        theta[:, :] = np.copy(theta_fiducial)
        theta[:, i] += h[i]
        sims_dash[:, :] = np.atleast_2d(
          simulator(theta[0], seed, simulator_args, sub_batch))
        d_dash = np.mean(sims_dash, axis = 0)
        dmudt[i, :] = (d_dash - d_fiducial) / h[i]
    return sims_dash, theta, dmudt
```
We initialise the compression module locally
```python
Compressor = score.Gaussian(len(data),
                            theta_fiducial,
                            prior_mean = prior_mean,
                            prior_covariance = prior_covariance)
```
### Derivative of the mean
First we will calculate the derivative of the mean. We need to push the derivative sub-batch calculator and the necessary variables for the function to the engines first.

The simulations are done asynchronously so that they can be sent off to the engines and the Jupyter kernel can still be used. To check if the calculations have finished one can run
```
derivative_data.ready()
```

We want to start by initialising some values on the engines so that we don't have to pass them each time. One of these (the size of the data) we're going to do by hand because I've not found a nice way to do this without running the simulator on each of the engines.
```python
print(Compressor.ndata)
```
```
> 105
```
We also want the number of simulations to calculate in a sub batch (which should always be 1 we think). We also want to get the amount of deviation of the fiducial parameters for the derivative.
```python
%%px --local
ndata = 105
h = abs(theta_fiducial)*0.05
sub_batch = 1
```
We then run the simulator on the engines
```python
n_sims_for_derivatives = 100
derivative_data = view.map_async(
    lambda sim_id: derivative_sub_batch_sims(sim_id),
    range(n_sims_for_derivatives))
```
Once the parallel simulations have finished then they can be retreived using
```
derivative_data.get()
```
We get collect the simulations and parameters and place them in the local Compressor module variables. We also calculate the derivative of the mean over the simulations.
```python
start = time.time()
Compressor.simulations = np.array([derivative_data.get()[i][0] for i in
    range(n_sims_for_derivatives)]).reshape(
    (n_sims_for_derivatives * sub_batch, data.shape[0]))
Compressor.parameters = np.array([derivative_data.get()[i][1] for i in
    range(n_sims_for_derivatives)]).reshape(
    (n_sims_for_derivatives * sub_batch, theta_fiducial.shape[0]))
Compressor.dmudt = np.mean([derivative_data.get()[i][2] for i in
    range(n_sims_for_derivatives)], axis = 0)
print(time.time()-start)
```
```
> 23.846494436264038
```
### Covariance at the fiducial parameter values
We now calculate the covariance and the mean at the fiducial parameter values after pushing the extra necessary parameters (and functions) that the engines need.
```python
n_sims_for_covariance = 1000
theta_range = [theta_fiducial for i in range(n_sims_for_covariance)]
mean_cov_data = view.map_async(
    lambda theta: sub_batch_sims(theta), theta_range)
```
Again, we can see whether the jobs have finished by running
```
mean_cov_data.ready()
```
We now calculate the mean, covariance and inverse covariance from these simulations and add the simulations to the local Compressor module variables.
```python
start = time.time()
mean_cov_sims = np.array([mean_cov_data.get()[i][0] for i in
    range(n_sims_for_covariance)]).reshape(
    (n_sims_for_covariance * sub_batch, data.shape[0]))
mean_cov_params = np.array([mean_cov_data.get()[i][1] for i in
    range(n_sims_for_covariance)]).reshape(
    (n_sims_for_covariance * sub_batch, theta_fiducial.shape[0]))
print(time.time()-start)
Compressor.mu = np.mean(mean_cov_sims, axis = 0)
Compressor.C = np.cov(mean_cov_sims, rowvar = False)
Compressor.Cinv = np.linalg.inv(Compressor.C)
Compressor.simulations = np.concatenate(
    [Compressor.simulations, mean_cov_sims])
Compressor.parameters = np.concatenate(
    [Compressor.parameters, mean_cov_params])
```
```
> 40.59565854072571
```
### Compute the Fisher matrix
We can calculate the Fisher information directly from the local Compressor module
```python
Compressor.compute_fisher()
```
### Define the compression function and compress the data
```python
def compressor(d, compressor_args):
    return Compressor.scoreMLE(d)
compressor_args = None

compressed_data = compressor(data, compressor_args)
```
## Make the density estimator
We only need a local density estimator, which can be run on a GPU for efficiency
```python
tf.reset_default_graph()
MDN = [ndes.MixtureDensityNetwork(n_inputs=5,
                                  n_outputs=5,
                                  n_components=3,
                                  n_hidden=[25,25],
                                  activations=[tf.tanh, tf.tanh],
                                  index = 1)]
```
## Create the Delfi instance for training the density estimator
```python
DelfiMDN = delfi.Delfi(compressed_data,
                       prior,
                       MDN,
                       Finv=Compressor.Finv,
                       theta_fiducial=theta_fiducial,
                       param_limits=[lower, upper],
                       param_names=['\Omega_m',
                                    'S_8',
                                    '\Omega_b',
                                    'h',
                                    'n_s'],
                       results_dir="simulators/cosmic_shear_map/results/mdn_")
```
Since no simulations are needed for Fisher pretraining, we do not need to edit this function.
```python
DelfiMDN.fisher_pretraining(patience = 20, batch_size = 100)
```
## Load simulations into Delfi
Since we run a number of simulations for the compression function, these can be preloaded into Delfi.
```python
compressed_sims = np.array([compressor(
    Compressor.simulations[i,:], compressor_args) for i in
    range(len(Compressor.simulations))])
DelfiMDN.load_simulations(compressed_sims, Compressor.parameters)
```
## Train Delfi
We are going to expand the sequential training so that we can do parallel simulations
```python
n_initial = 250
n_batch = 250
n_populations = 23
safety = 5
plot = True
batch_size = 100
validation_split = 0.1
epochs = 300
patience = 20
save_intermediate_posteriors = True

for population in tqdm.tnrange(n_populations + 1, desc = "Populations"):

    if population == 0:
        proposal = priors.TruncatedGaussian(DelfiMDN.theta_fiducial,
                                            9*DelfiMDN.Finv,
                                            DelfiMDN.lower,
                                            DelfiMDN.upper)
        ps = proposal.draw(to_draw = safety * n_initial)
        n_sims_to_run = n_initial
    else:
        DelfiMDN.proposal_samples = DelfiMDN.emcee_sample(
          DelfiMDN.log_geometric_mean_proposal,
          [DelfiMDN.proposal_samples[-j,:] for j in range(DelfiMDN.nwalkers)],
          main_chain=DelfiMDN.proposal_chain_length)
        ps_batch = DelfiMDN.proposal_samples[-safety * n_batch:,:]
        n_sims_to_run = n_batch

    calculated = 0
    first = 0
    tries = 0
    last = n_sims_to_run
    while calculated < n_sims_to_run and last < safety * n_sims_to_run:
        batch = view.map_async(
          lambda theta: sub_batch_sims(theta), ps[first: last])
        batch_list = [batch.get()[i] for i in
          range(len(batch.get())) if batch.get()[i] is not None]
        calculated += len(batch_list)
        first = last
        last = last + (n_sims_to_run - calculated)
        if calculated > 0:
            DelfiMDN.load_simulations(np.array([compressor(batch_list[i][0], compressor_args) for i in range(len(batch_list))]), np.array([batch_list[i][1] for i in range(len(batch_list))]))
            #DelfiMDN.ps = np.concatenate([DelfiMDN.ps, (np.array([batch_list[i][1] for i in range(len(batch_list))]) - DelfiMDN.theta_fiducial) / DelfiMDN.fisher_errors])
            #DelfiMDN.xs = np.concatenate([DelfiMDN.xs, (np.array([compressor(batch_list[i][0], compressor_args) for i in range(len(batch_list))]) - DelfiMDN.theta_fiducial) / DelfiMDN.fisher_errors])
        tries += 1
    if last == safety * n_sims_to_run:
        print("failed too often. only using " + str(calculated) + " sims instead of " + str(n_sims_to_run))

    #DelfiMDN.n_sims = DelfiMDN.ps.shape[0]
    #val_loss, train_loss = DelfiMDN.trainer.train(DelfiMDN.sess, [DelfiMDN.ps.astype(np.float32), DelfiMDN.xs.astype(np.float32)], validation_split = validation_split, epochs = epochs, batch_size=max(DelfiMDN.n_sims//8, batch_size), progress_bar=DelfiMDN.progress_bar, patience=patience, saver_name='{}tmp_model'.format(DelfiMDN.results_dir), save_during_early_stopping = True)

    #DelfiMDN.training_loss = np.concatenate([DelfiMDN.training_loss, train_loss])
    #DelfiMDN.validation_loss = np.concatenate([DelfiMDN.validation_loss, val_loss])
    #DelfiMDN.sequential_training_loss.append(train_loss[-1])
    #DelfiMDN.sequential_validation_loss.append(val_loss[-1])
    #DelfiMDN.sequential_nsims.append(DelfiMDN.n_sims)
    DelfiMDN.train_ndes(training_data=[DelfiMDN.x_train, DelfiMDN.y_train], batch_size=max(DelfiMDN.n_sims//8, batch_size), validation_split=validation_split, epochs=epochs, patience=patience, saver_name=None)
    DelfiMDN.stacked_sequential_training_loss.append(np.sum(np.array([DelfiMDN.training_loss[n][-1]*DelfiMDN.stacking_weights[n] for n in range(DelfiMDN.n_ndes)])))
    DelfiMDN.stacked_sequential_validation_loss.append(np.sum(np.array([DelfiMDN.validation_loss[n][-1]*DelfiMDN.stacking_weights[n] for n in range(DelfiMDN.n_ndes)])))
    DelfiMDN.sequential_nsims.append(DelfiMDN.n_sims)

    # Generate posterior samples
    print('Sampling approximate posterior...')
    DelfiMDN.posterior_samples = DelfiMDN.emcee_sample(DelfiMDN.log_posterior, [DelfiMDN.posterior_samples[-i,:] for i in range(DelfiMDN.nwalkers)], main_chain=DelfiMDN.posterior_chain_length)

    # Save posterior samples to file
    if save_intermediate_posteriors:
        f = open('{}posterior_samples_{:d}.dat'.format(DelfiMDN.results_dir, population), 'w')
    else:
        f = open('{}posterior_samples.dat'.format(DelfiMDN.results_dir), 'w')
    np.savetxt(f, DelfiMDN.posterior_samples)
    f.close()

    np.savez('{}parameters_summaries.npz'.format(DelfiMDN.results_dir), parameters = DelfiMDN.ps, summaries = DelfiMDN.xs)
    print('Done.')

    # If plot == True, plot the current posterior estimate
    if plot == True:
        DelfiMDN.triangle_plot([DelfiMDN.posterior_samples], savefig=True, filename='{}seq_train_post_{:d}.pdf'.format(DelfiMDN.results_dir, population))
        DelfiMDN.sequential_training_plot(savefig=True, filename='{}seq_train_loss.pdf'.format(DelfiMDN.results_dir))
```
