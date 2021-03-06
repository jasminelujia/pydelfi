import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.interpolate as interpolate
from scipy.stats import norm
from scipy.special import jv
from scipy.stats import wishart
from scipy.stats import norm as normal
from scipy.stats import multivariate_normal
import pickle
import scipy.integrate as integrate
from .cosmology import *
import scipy.constants as sc
import healpy as hp
import numpy.random as npr
import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# Compute the data vector
def power_spectrum(theta, pz, l_min, l_max):
    
    # Process args
    nz = len(pz)

    # Evaluate the required (derived) cosmological parameters
    omm = theta[0]
    sigma8 = theta[1]*np.sqrt(0.3/theta[0])
    omb = theta[2]
    h = theta[3]
    ns = theta[4]
    omde = 1.0 - omm
    omnu = 0
    omk = 0
    hubble = h*100
    w0 = -1.
    wa = 0
    
    # Initialize cosmology object
    cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

    # Numerics parameters
    zmax = 2
    rmax = cosmo.a2chi(z2a(zmax))
    power_zpoints = int(np.ceil(5*zmax))
    power_kpoints = 200
    distance_zpoints = int(np.ceil(10*zmax))
    wpoints = int(np.ceil(15*zmax))
    kmax = 10
    clpoints = 2**7 + 1

    # Compute the matter power spectrum at the cosmology
    z = np.linspace(0, zmax, power_zpoints)
    logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
    logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))
    
    # 2D linear interpolator for P(k;z)
    logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

    # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
    zvalues = np.linspace(0, zmax, distance_zpoints)
    rvalues = np.zeros((len(zvalues)))

    # Perform integration to compute r(z) at specified points according to cosmology
    for i in range(0, len(zvalues)):
        rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i])

    # Generate interpolation functions to give r(z) and z(r) given cosmology
    r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
    z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

    # Set the maximum comoving distance corresponding to the maximum redshift
    rmax = rvalues[-1]

    # Compute lensing weights...

    w = []
    
    # Compute the weight function associated with each bin in turn, over r-points and then interpolate
    for i in range(0, nz):

        # r-points to evaluate weight at before interpolation
        rpoints = np.linspace(0, rmax, wpoints)

        # Initialize weights
        weight = np.zeros(wpoints)

        # Compute integral for the rest of the points
        for j in range(1, wpoints):
            x = np.linspace(rpoints[j], rmax, 2**6 + 1)
            dx = x[1] - x[0]
            intvals = rpoints[j]*pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
            weight[j] = integrate.romb(intvals, dx)

        # Interpolate (generate interpolation function) and add interpolation function to the array w
        interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
        w.append(interp)
    
    # Tensor for cls
    n_l = l_max - l_min + 1
    cls = np.zeros((nz, nz, n_l))

    # Pull required cosmological parameters out of cosmo
    r_hubble = sc.c/(1000*hubble)
    A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

    # Compute Cls
    for k in range(n_l):
        l = l_min + k
        rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
        dr = rs[1] - rs[0]
        for i in range(0, nz):
            for j in range(i, nz):
                intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                cls[i, j, k] = integrate.romb(intvals, dr)
                cls[j, i, k] = cls[i, j, k]
        #cls[:,:,L] = cls[:,:,L] + N
                
    return cls

def a_lm_r_to_c(a_lm_r):

    # convert real a_lm array to complex
    n_fields, n_a_lm_r = a_lm_r.shape
    el = (n_a_lm_r - 1) // 2
    a_lm_c = np.zeros((n_fields, int(el + 1)), 'complex')
    a_lm_c[:, 0] = a_lm_r[:, el] + 0.0j
    for m in range(1, el + 1):
        a_lm_c[:, m] = a_lm_r[:, el + m] / np.sqrt(2.0) - \
                       a_lm_r[:, el - m] / np.sqrt(2.0) * 1.0j
    return a_lm_c

def cl_to_cl_hat(cls, l_min, l_max, sig_n_p, mask, seed=None):

    # setup
    n_l = l_max - l_min + 1
    n_a_lm_c = int((l_max + 1) * (l_max + 2) / 2)
    n_fields, n_pix = mask.shape
    n_side = hp.npix2nside(n_pix)
    
    # draw some a_lms
    a_lm_c = np.zeros((n_fields, n_a_lm_c), 'complex')
    if seed is not None:
        npr.seed(seed)
    for i in range(0, n_l):

        # generate 2l+1 real a_lms and convert to l+1 complex a_lms
        # with appropriate variance. use Healpy/C++ indexing. calculate 
        # observed covariance for score compression
        el = l_min + i
        ems = np.arange(el + 1)
        cov_chol = np.linalg.cholesky(cls[:, :, i])
        a_lm_r = np.dot(cov_chol, npr.randn(n_fields, 2 * el + 1))
        inds = ems * (2 * l_max + 1 - ems) // 2 + el
        a_lm_c[:, inds] = a_lm_r_to_c(a_lm_r)

    # project to map and add pixel noise and mask, then transform back to alm
    field = np.zeros((n_fields, n_pix))
    for i in range(0, n_fields):
        #with suppress_stdout_stderr():
        field[i, :] = (hp.alm2map(a_lm_c[i, :], n_side, verbose=False) + \
                       npr.randn(n_pix) * np.sqrt(sig_n_p[i,:])) * mask[i, :]
        a_lm_c[i, :] = hp.map2alm(field[i, :], lmax=l_max, mmax=l_max)

    # compute observed power spectra
    c_l_hat = np.zeros((n_fields, n_fields, n_l))
    for i in range(n_fields):
        for j in range(i + 1):
            c_l_hat[i, j, :] = hp.alm2cl(alms1 = a_lm_c[i, :], alms2 = a_lm_c[j, :], \
                                          lmax=l_max, mmax=l_max)[l_min:]
            c_l_hat[j, i, :] = c_l_hat[i, j, :]

    return c_l_hat

def cl_to_maps(cls, l_min, l_max, mask):

    # setup
    n_l = l_max - l_min + 1
    n_a_lm_c = int((l_max + 1) * (l_max + 2) / 2)
    n_fields, n_pix = mask.shape
    n_side = hp.npix2nside(n_pix)
    
    # draw some a_lms
    a_lm_c = np.zeros((n_fields, n_a_lm_c), 'complex')
    for i in range(0, n_l):

        # generate 2l+1 real a_lms and convert to l+1 complex a_lms
        # with appropriate variance. use Healpy/C++ indexing. calculate
        # observed covariance for score compression
        el = l_min + i
        ems = np.arange(el + 1)
        cov_chol = np.linalg.cholesky(cls[:, :, i])
        a_lm_r = np.dot(cov_chol, npr.randn(n_fields, 2 * el + 1))
        inds = ems * (2 * l_max + 1 - ems) // 2 + el
        a_lm_c[:, inds] = a_lm_r_to_c(a_lm_r)

    # project to map and add pixel noise and mask, then transform back to alm
    field = np.zeros((n_fields, n_pix))
    for i in range(0, n_fields):
        #with suppress_stdout_stderr():
        field[i, :] = hp.alm2map(a_lm_c[i, :], n_side, verbose=False)

    return field

def maps_to_cl_hat(field, l_min, l_max, sig_n_p, mask):

    # setup
    n_l = l_max - l_min + 1
    n_a_lm_c = int((l_max + 1) * (l_max + 2) / 2)
    n_fields, n_pix = mask.shape
    n_side = hp.npix2nside(n_pix)
    a_lm_c = np.zeros((n_fields, n_a_lm_c), 'complex')

    # project to map and add pixel noise and mask, then transform back to alm
    noisy_masked_field = np.zeros((n_fields, n_pix))
    for i in range(0, n_fields):
        #with suppress_stdout_stderr():
        noisy_masked_field[i, :] = (field[i, :] + \
                       npr.randn(n_pix) * np.sqrt(sig_n_p[i,:])) * \
                       mask[i, :]
        a_lm_c[i, :] = hp.map2alm(noisy_masked_field[i, :], lmax=l_max, mmax=l_max)

    # compute observed power spectra
    c_l_hat = np.zeros((n_fields, n_fields, n_l))
    for i in range(n_fields):
        for j in range(i + 1):
            c_l_hat[i, j, :] = hp.alm2cl(alms1 = a_lm_c[i, :], alms2 = a_lm_c[j, :], \
                                          lmax=l_max, mmax=l_max)[l_min:]
            c_l_hat[j, i, :] = c_l_hat[i, j, :]

    return c_l_hat


def simulate(theta, seed, simulator_args, batch):
    
    # Set the seed
    np.random.seed(seed)
    
    pz_fid = simulator_args[0]
    l_min = simulator_args[1]
    l_max = simulator_args[2]
    sig_n_p = simulator_args[3]
    mask = simulator_args[4]
    n_ell_bins = simulator_args[5]
    nz = len(pz_fid)
    
    # Photo-z parameters
    #z = np.linspace(0, pz_fid[0].get_knots()[-1], len(pz_fid[0].get_knots()))
    #pz_new = [0]*nz
    #for i in range(nz):
    #    p = pz_fid[i](z+theta[5+i])
    #    p = p/np.trapz(p, z)
    #    pz_new[i] = interpolate.InterpolatedUnivariateSpline(z, p, k=3)
    #pz = pz_new
    pz = pz_fid
    
    # Compute theory power spectrum
    C = power_spectrum(theta, pz, l_min, l_max)
    
    # Create realization of field
    field = cl_to_maps(C, l_min, l_max, mask)
    
    sims = np.zeros((batch, n_ell_bins*nz*(nz+1)//2))
    for i in range(batch):
    
        # Realize noisy power spectrum
        C_hat = maps_to_cl_hat(field, l_min, l_max, sig_n_p, mask)

        # Bin up the power spectra
        l = np.linspace(l_min, l_max, l_max-l_min+1)
        bin_edges = np.linspace(l_min, l_max, n_ell_bins+1)

        d = np.array([])
        for k in range(n_ell_bins):
            l_selection = (l >= bin_edges[k])*(l < bin_edges[k+1])
            C_hat_k = np.mean(C_hat[:,:,l_selection], axis=-1)
            x = np.tril(C_hat_k).flatten()
            ind = np.nonzero(x)
            d = np.concatenate([d, x[ind]])
        sims[i,:] = d
    return sims
