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

# Compute the data vector
def power_spectrum(theta, pz, modes, N):
    
    # Unpack sim args
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
        rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

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
    cls = np.zeros((nz, nz, len(modes)))

    # Pull required cosmological parameters out of cosmo
    r_hubble = sc.c/(1000*hubble)
    A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

    # Compute Cls
    for L in range(len(modes)):
        l = modes[L]
        rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
        dr = rs[1] - rs[0]
        for i in range(0, nz):
            for j in range(i, nz):
                intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                cls[i, j, L] = integrate.romb(intvals, dr)
                cls[j, i, L] = cls[i, j, L]
        cls[:,:,L] = cls[:,:,L] + N
                
    return cls

def simulate(theta, sim_args):
    
    pz_fid, modes, N, nl = sim_args
    
    nz = int(len(pz_fid))
    nmodes = len(modes)
    pz = pz_fid
    
    # simulated data holder
    d = np.zeros((int(nmodes*nz*(nz+1)/2)))
    
    # Compute theory power spectrum
    C = power_spectrum(theta, pz, modes, N)
    
    # Realize noisy power spectrum
    C_hat = np.zeros((nz, nz, nmodes))
    for i in range(nmodes):
        C_hat[:, :, i] = wishart.rvs(df=nl[i], scale=C[:,:,i])/nl[i]
        x = np.tril(C_hat[:, :, i])
        x = x[np.nonzero(x)]
        d[i*int(nz*(nz+1)/2):(i+1)*int(nz*(nz+1)/2)] = x
    
    return d

def fisher_matrix(Cinv, dCdt, npar, nl, Qinv):
    
    F = np.zeros((npar, npar))
    for a in range(len(dCdt)):
        for b in range(len(dCdt)):
            for l in range(len(Cinv[0,0,:])):
                F[a,b] += 0.5*nl[l]*np.trace( np.dot(Cinv[:,:,l], np.dot(dCdt[a,:,:,l], np.dot(Cinv[:,:,l], dCdt[b,:,:,l]) ) )) 
    F = F + Qinv
    Finv = np.linalg.inv(F)
    fisher_errors = np.sqrt(np.diag(Finv))
    return F, Finv, fisher_errors

def score(d, score_args):
    
    Finv, theta_fiducial, prior_mean, Qinv, Cinv, dCdt, nl = score_args
    
    # Compute the score
    t = np.zeros(len(Finv))
    for a in range(len(Finv)):
        for l in range(len(Cinv[0,0,:])):
            t[a] += nl[l]*(-0.5*np.trace(np.dot(Cinv[:,:,l], dCdt[a,:,:,l])) + 0.5*np.trace(np.dot( np.dot(Cinv[:,:,l], np.dot(dCdt[a,:,:,l], Cinv[:,:,l])), d[:,:,l]) ) )
   
    # Make it an MLE
    t = np.dot(Finv, t) + theta_fiducial + np.dot(Finv, np.dot(Qinv, prior_mean - theta_fiducial))
    
    # Return re-scaled statistics
    return t
