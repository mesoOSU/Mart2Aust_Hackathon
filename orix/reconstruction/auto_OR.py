# -*- coding: utf-8 -*-
import warnings

from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Symmetry, Misorientation
from orix.quaternion.rotation import Rotation
from orix.quaternion.orientation import Orientation
from orix.quaternion.orientation_region import OrientationRegion

from scipy import optimize
from scipy.io import loadmat # FOR MATLAB .mat FILE TESTING

import yardleyVariants # Fix this import later!

# Test variables for function eval against MATLAB
#x = np.array([0.1, 0.5, 3], dtype=float)
#ksi_sample = np.array([2.9, 7.7, 8.2], dtype=float)
#mu = np.array([0.2, 0.4, 0.6], dtype=float)
#sigma = np.array([1, 1.5, 2], dtype=float)
OR = np.array([2.16, 8.06, 8.30], dtype=float)

# Test construction of the 'ebsd' orientation object
# using pre-built functions of orix
# Start by loading in the test 'AF96.ang' file
euler1, euler2, euler3, x, y, iq, ci, phase_id = np.loadtxt("AF96.ang", unpack=True)[0:8,:]

# Extract out the Euler angles and assemble rotations
euler_angles = np.column_stack((euler1, euler2, euler3))
rotations = Rotation.from_euler(euler_angles, direction='MTEX')

# # Handle the conversion of Euler to the spatial ref. frame
# # how MTEX does it.
# xvector = np.array([1, 0, 0], dtype=float)
# yvector = np.array([0, 1, 0], dtype=float)
# conversion = Rotation.from_axes_angles(xvector-yvector, np.pi)

# # Convert everything to rotation matrix to perform the multiplication
# # and then convert back into the quats
# conv_rmat = conversion.to_matrix()
# rot_rmat  = rotations.to_matrix()
# rotations2 = Rotation.from_matrix(conv_rmat * rot_rmat) # Not working...


# Define the structure of the phases
structures = [
    Structure(
        title="unindexed"
    ),
    Structure(
        title="austenite",
        atoms=[Atom("fe", [0] * 3)],
        lattice=Lattice(0.365, 0.365, 0.365, 90, 90, 90)
    ),
    Structure(
        title="martensite",
        atoms=[Atom("fe", [0] * 3)],
        lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
    )]
# Assemble into a PhaseList
pl = PhaseList(space_groups=[1, 225, 225], structures=structures)

# Assemble into a CrystalMap
ebsd = CrystalMap(
    rotations=rotations,
    phase_id=phase_id,
    x=x,
    y=y,
    phase_list=pl,
    scan_unit='um'
)




### Below are functions that are unfinished skeleton ports from MATLAB equivalents
### Complete functionality requires porting of MTEX's unimodalODF, eval, and others
### along with a better porting of the orientation objects of MTEX.

def calculateOR(ebsd):
    """ Skeleton version of calculateOR ported from
    MATLAB.
    
    Input 'ebsd' should be an orix CrystalMap structure
    Inputs 'options' will be a python dict loaded in from 'load_options'
    
    Returns 'ksi_optim' and 'halfwidth_optim' which should be numpy arrays?
    """
    
    # Extract the phase symmetries and set the
    # sample symmetry directly
    CS_HT = ebsd['austenite'].phases[1].point_group
    CS_LT = ebsd['martensite'].phases[1].point_group
    SS    = Symmetry((1, 0, 0, 0)) # Triclinic Symmetry object
    
    # Search the low temp. phase for an acceptably close PAG
    searchable_ebsd = ebsd['martensite']
    
    for iteration in range (0, 10): # This iteration is for the grain-cut not optimization
        # Search all unused areas
        # aus_grain_ids, _, ausor = HT_grain_guess(searchable_ebsd)
        # martensite = searchable_ebsd.orientations[aus_grain_ids] # Be careful of 0-indexing
        # The above should retain the martensite symmetry, but what about lattice info?

        # Perform the downsampling per options values
        
        #   code?
        
        # Generate an initial guess for parameters to be estimated
        ksi_initial = np.array([5.26, 10.3, 10.53])
        halfwidth_initial = 2.5
        
        # Check that the ksi values are reasonable
        #flag = calc_T2R(ksi_initial, CS_HT, CS_LT)[1] # May have to be removed as Yardley isnt returning a flag
        #if flag:
        #    warnings.warn('Non-physical initial ksi angles!')
        
        # Set prior parameters based on rough estimates from
        # Yardley and Payton 2014 conference paper
        ksi_prior_mu    = np.array([5, 9, 10])
        ksi_prior_sigma = np.array([2, 2, 2])
        
        # Noise is modeled as a unimodal odf centered on the cube orientation.
        # Parameter to be estimated by Bayesian inference is the halfwidth of the
        # odf kernel. Halfwidth distribution is assumed folded Gaussian. This
        # approximates uniform from 0->~1 degree then decaying at larger noise values
        halfwidth_prior_mu = 1
        halfwidth_prior_sigma = 2
       
        # Put together a prior_pars dict here
        prior_pars = {
            'ksi_prior_mu': ksi_prior_mu,
            'ksi_prior_sigma': ksi_prior_sigma,
            'halfwidth_prior_mu': halfwidth_prior_mu,
            'halfwidth_prior_sigma': halfwidth_prior_sigma,            
            'CS_A': CS_HT,
            'CS_M': CS_LT,
            'SS': SS
            }
        
        # MAP estimate of parameters via optimization
        # MAP_options = 
        optimfunc = lambda samples: -posterior_pdf_fminfunc(samples, prior_pars, martensite)
        x0 = np.array([ksi_initial, halfwidth_initial]) # This may not be happy. what about degree on halfwidth?
        
        # Ensure the constraint that ksi_1 < ksi_2 and ksi_3.
        initial_guess = 0
        cnt = 0
        while (initial_guess == 0):
            cnt += cnt
            
            MAPpars = optimize.fmin(func=optimfunc, x0=x0, xtol=0.0001, ftol=0.0001, full_output=True)[0]
            
            if (MAPpars[0] < MAPpars[1] and MAPpars[2] or cnt > 2):
                initial_guess = 1
                
                # Enforce the constraint if need be
                if (MAPpars[1] > MAPpars[2]):
                    MAPpars[1] = MAPpars[2] - 1e-3
        
        # Return the optimized solution           
        ksi_optim = MAPpars[0:3]
        halfwidth_optim = MAPpars[3]
        
        # Add logic here to exit the for loop if needed?
        
    return ksi_optim, halfwidth_optim

def posterior_pdf_fminfunc(samples, prior_pars, martensite):
    """ This is our anon. function that takes in 'samples' which is a numpy array (4,),
    'prior_pars' which is a standard dict, and 'martensite' which is a phase-specific
    orix crystalMap structure?
    
    Returns the maximized posterior value in 'p'
    
    Note: This function require calc_T2R, global_pole_figure_estimation, and 
    martensite_posterior_log_likelihood to work correctly...
    """
    # Extract out the individual ksi and halfwidth values
    ksi1 = np.abs(samples[0])
    ksi2 = np.abs(samples[1])
    ksi3 = np.abs(samples[2])
    halfwidth = np.abs(samples[3]) # degree
    
    OR = np.array[ksi1, ksi2, ksi3]
    T2R = calc_T2R(OR, prior_pars['CS_A'], prior_pars['CS_M'])
    
    austenite = Misorientation.equivalent(martensite.orientations) * T2R
    
    austenite_proposal = global_pole_figure_estimation(austenite, \
                                                       austenite['austenite'].phases[1].point_group, \
                                                       Symmetry((1, 0, 0, 0)), \
                                                       1)
    # I'm not sure if the austenite['austenite'] makes sense above here...
    
    temp = np.empty(austenite_proposal.orientations.size, dtype=float)
    for kk in range(0, austenite_proposal.orientations.size-1):
        temp[kk] = martensite_posterior_log_likelihood(martensite, OR, halfwidth, \
                                                       austenite_proposal.orientations[kk], prior_pars)
    
    id = np.argwhere(temp == np.max(temp))
    p = temp[id] # Is this correct indexing?
    
    return p
    
def martensite_posterior_log_likelihood(martensite, OR, halfwidth, austenite, prior_pars):
    """ Straight-forward computation of the log-likelihood used by posterior_pdf_fminfunc

    Input martensite should be an orix crystalMap?
    Input OR is a numpy array (3,)
    Input halfwidth is a scalar
    Input austenite is an orix orientation
    Input prior_pars is a standard dict
    
    Output log-like is an array?
    
    Note: This function require martensite_likelihood to work properly. The functions
    ksi_prior and halfwidth_prior are ported and functioning!
    """    
    data_log_like = np.sum(np.log( \
        martensite_likelihood(martensite, OR, halfwidth, austenite, prior_pars['CS_A'], prior_pars['CS_M']) \
        ))
    
    ksi_log_prior_prob = np.sum(np.log( \
        ksi_prior(OR, prior_pars['ksi_prior_mu'],prior_pars['ksi_prior_sigma']) \
        ))
        
    hw_log_prior_prob = np.sum(np.log( \
        halfwidth_prior(halfwidth, prior_pars['halfwidth_prior_mu'],prior_pars['halfwidth_prior_sigma']) \
        ))
    
    log_like = data_log_like + ksi_log_prior_prob + hw_log_prior_prob
    
    return log_like        
        
def martensite_likelihood(martensite, OR, halfwidth, austenite, cs_a, cs_m):
    """ Computes the likelihood of measuring martensite orientations for a given
    set of parameters
    
    Note: This function require unimodalODF and MTEX's eval to work properly.
    """
    austenite_odf = unimodalODF(austenite, cs_a, cs_m, hw=halfwidth)
    
    T2R = calc_T2R(OR, cs_a, cs_m)
    
    pot_aus = Misorientation.equivalent(martensite.orientations) * T2R
    
    l = MTEXeval(austenite_odf, pot_aus)
    l[np.argwhere(l < 1e-6)] = 1e-6
    l = np.reshape(l, (24, l.size()/24)).transpose() # Direct port not tested
    
    likelihood = np.sum(l, axis=1) * 0.057 # Sum the columns
    
    return likelihood # Should be size (l,)
    
def global_pole_figure_estimation(austenite, aust_sym, samp_sym, flag):
    """ comments
    """
    




### Below are functions that are PARTIALLY tested and working against MATLAB
### but require additional work to complete

def calc_T2R(OR, cs_r, cs_t):
    """ Returns the T2R misorientation object using the first rotation 
    matrix returned from YardleyVariants. 
    
    Input OR is a numpy array (3,) containing the ksi values
    Input cs_r is an orix equiv. crystalSymmetry class from MTEX for the austenite
    Input cs_t is an orix equiv. crystalSymmetry class from MTEX for the martensite
    
    Output will be T2R, an orix equiv. Misorientation class from MTEX?
    """
    
    if (np.size(OR) == 3):
        OR = yardleyVariants.yardley_variants(OR) # We are no longer returning a flag?

    var1 = Rotation.from_matrix(OR[0, 0:, 0:]) # Grab the first variant and cast as rotation
    # Note that this "var1" quat matches T2R output from MATLAB, but there is no
    # inherent misorientation information from Aust to Mart?

    # Create the T2R misorientation object? Orix has no sample symmetry so we may
    # need to build a Symmetry.from_generators(cs_r, cs_t) type approach
    # to define the symmetry to build the T2R misorientation object?

    return var1 # Should be T2R later




### Below are functions that are tested and working against MATLAB

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
