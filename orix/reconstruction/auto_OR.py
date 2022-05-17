# -*- coding: utf-8 -*-
import warnings

import numpy as np

# from scipy.io import loadmat 3 FOR MATLAB .mat FILE TESTING

# Test variables for function eval against MATLAB
x = np.array([0.1, 0.5, 3], dtype=float)
ksi_sample = np.array([2.9, 7.7, 8.2], dtype=float)
mu = np.array([0.2, 0.4, 0.6], dtype=float)
sigma = np.array([1, 1.5, 2], dtype=float)

def fldnrmPDF(x, mu, sigma):
    """ Computes the probability density function of the
    folded normal distribution.
        
    References:
        [1] FC Leone, LS Nelson, RB Nottingham, Technometrics 3 (1961) 543.
        [2] RC Elandt, Technometrics 3 (1961) 551.
            
    """
    # Ensure that inputs are all numpy 1D arrays for computation
    # even if the input values are just scalars!
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    
    a = 2 * sigma**2
    
    f = np.empty(x.size, dtype=float)
    f = ( 1 / ( np.sqrt( 2 * np.pi ) * sigma ) ) * \
        (np.exp(-(x-mu)**2/a) + np.exp(-(x+mu)**2/a))
        
    return f

def ksi_prior(ksi_sample, mu, sigma):
    """ Computes the prior distribution on the orientation
    relationship.
    
    All inputs are assumed to be of size (3,)
    """
    # There should be error handling here for shape assertions
    p = np.empty(ksi_sample.size, dtype=float)
    
    if np.size(np.argwhere(ksi_sample < 0)) > 0:
        p = np.ones(p.size, dtype=float) * 1e-100
    elif (ksi_sample[0] > ksi_sample[1]):
        p = np.ones(p.size, dtype=float) * 1e-100
    else:
        for i in range (0, 2):
            p[i] = fldnrmPDF(ksi_sample[i],mu[i],sigma[i])
    
    return p


def halfwidth_prior(halfwidth_sample, mu, sigma):
    """ comments
    
    """
    # Do we want to just let fldnrmPDF handle the assertions?
    
    p = fldnrmPDF(halfwidth_sample, mu, sigma)
    
    return p

