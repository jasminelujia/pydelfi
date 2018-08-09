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
def power_spectrum(theta, sim_args):
    
    # Unpack sim args
    pz = sim_args[0]
    modes = sim_args[1]
    N = sim_args[2]
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
    cls = np.zeros((len(modes), nz, nz))

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

# Compute the data vector
def d_power_spectrum_db(theta, sim_args):
    
    # Unpack sim args
    pz = sim_args[0]
    modes = sim_args[1]
    N = sim_args[2]
    pzprime = sim_args[3]
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
    dwdb = []

    # Compute the weight function associated with each bin in turn, over r-points and then interpolate
    for i in range(0, nz):

        # r-points to evaluate weight at before interpolation
        rpoints = np.linspace(0, rmax, wpoints)

        # Initialize weights
        weight = np.zeros(wpoints)
        dweightdb = np.zeros(wpoints)
        # Compute integral for the rest of the points
        for j in range(1, wpoints):
            x = np.linspace(rpoints[j], rmax, 2**6 + 1)
            dx = x[1] - x[0]
            intvals = rpoints[j]*pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
            weight[j] = integrate.romb(intvals, dx)
            
            x = np.linspace(rpoints[j], rmax, 2**6 + 1)
            dx = x[1] - x[0]
            intvals = rpoints[j]*pzprime[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
            dweightdb[j] = integrate.romb(intvals, dx)

        # Interpolate (generate interpolation function) and add interpolation function to the array w
        interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
        w.append(interp)
        interp = interpolate.InterpolatedUnivariateSpline(rpoints, dweightdb, k = 3)
        dwdb.append(interp)
    
    # Tensor for cls
    cls = np.zeros((len(modes), nz, nz))

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
                intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)* (w[i](rs)*dwdb[j](rs) + dwdb[i](rs)*w[j](rs)) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                cls[i, j, L] = integrate.romb(intvals, dr)
                cls[j, i, L] = cls[i, j, L]
        cls[:,:,L] = cls[:,:,L] + N
                
    return cls

def simulate(theta, sim_args):
    
    pz_fid = sim_args[0]
    modes = sim_args[1]
    N = sim_args[2]
    nl = sim_args[3]
    nz = len(pz_fid)
    nmodes = len(modes)
    
    # Photo-z parameters
    z = np.linspace(0, pz_fid[0].get_knots()[-1], len(pz_fid[0].get_knots()))
    pz_new = [0]*nz
    for i in range(nz):
        p = pz_fid[i](z+theta[5+i])
        p = p/np.trapz(p, z)
        pz_new[i] = interpolate.InterpolatedUnivariateSpline(z, p, k=3)
    pz = pz_new
    
    # Compute theory power spectrum
    C = power_spectrum(theta, [pz, modes, N])
    
    # Realize noisy power spectrum
    C_hat = np.zeros((nz, nz, nmodes))
    for i in range(nmodes):
        C_hat[:, :, i] = wishart.rvs(df=nl[i], scale=C[:,:,i])/nl[i]
    
    return C_hat

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


def projected_score(d, projection_args):
    
    Finv = projection_args[0]
    P = projection_args[1]
    theta_fiducial = projection_args[2]
    fisher_errors = projection_args[3]
    prior_mean = projection_args[4]
    Qinv = projection_args[5]
    Cinv = projection_args[6]
    dCdt = projection_args[7]
    modes = projection_args[8]
    nl = projection_args[9]
    
    # Compute the score
    dLdt = np.zeros(len(Finv))
    for a in range(len(Finv)):
        for l in range(len(modes)):
            dLdt[a] += nl[l]* (-0.5*np.trace(np.dot(Cinv[:,:,l], dCdt[a,:,:,l])) + 0.5*np.trace(np.dot( np.dot(Cinv[:,:,l], np.dot(dCdt[a,:,:,l], Cinv[:,:,l])), d[:,:,l]) ) )

    # Do the projection
    t = np.zeros(len(P))
    for a in range(len(P)):
        t[a] = dLdt[a] - np.dot(P[a], dLdt[len(P):])   
    
    # Make it an MLE
    t = np.dot(Finv[0:len(P), 0:len(P)], t) + theta_fiducial[0:len(P)] + np.dot(Finv[:len(P),:len(P)], np.dot(Qinv[:len(P),:len(P)], prior_mean[:len(P)] - theta_fiducial[:len(P)]))
    
    # Return re-scaled statistics
    return t#(t - theta_fiducial[0:len(P)])/fisher_errors[0:len(P)]

def simulationABC(theta, simABC_args):
    
    # Unpack the sim_args
    sim_args = simABC_args[0]
    projection_args = simABC_args[1]
    prior_args = simABC_args[2]
    
    # Draw nuisance parameters from the prior
    nuisances = truncated_gaussian_prior_draw(prior_args)[len(projection_args[1]):]
    
    # Simulate data
    d = simulate(np.concatenate([theta, nuisances]), sim_args)
    
    # Compute the projected score
    t = projected_score(d, projection_args)
    
    return t

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

def truncated_gaussian_prior(theta, prior_args):
    
    mean = prior_args[0]
    C = prior_args[1]
    lower = prior_args[2]
    upper = prior_args[3]
    
    return flat_prior(theta, [lower, upper])*multivariate_normal.pdf(theta, mean=mean, cov=C)
