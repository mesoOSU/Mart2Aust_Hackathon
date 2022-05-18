# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:19:23 2022

@author: Johnny Cales
"""

import math
import numpy
import scipy.special as sc
from orix.quaternion import Rotation, \
    Orientation as phil  # Name doesn't have to be 'Phil', just a name that R can call upon!


# g as Orix Orientation Object, L is a scalar bandwidth, Psi
# is kernel, and weights is a double vector same length as g
def calc_ODF(g, L, Psi, weights):
    # get orix R rotations from g
    # Has not been confirmed to work!
    # g: Array of orientation objects
    R = phil.orientation(g)

    # use Orix to_euler to get [phi1, Phi, phi2] from R
    # Has not been confirmed to work!
    # result = to_euler(R)
    Rotations = phil.to_euler(R)
    phi1 = Rotations[0]
    Phi = Rotations[1]
    phi2 = Rotations[2]

    # convert [phi1, Phi, phi2] to [[alpha beta gamma]]
    # Has not been confirmed to work!
    alpha = phi1 - math.pi / 2
    beta = Phi
    gamma = phi2 - (3 / 2) * math.pi

    # export A from Psi % export Chebyshev coefficients
    # Has not been confirmed to work!
    # Ported from MATLAB from 'calcFourier.m' (Line 21)
    # A = component.psi.A;
    # A = A(1:min(max(2,L+1),length(A)));
    A = Psi.A
    result = min(max(2, L + 1), len(A))
    A = A[1:result]

    # calculate c from weights
    # Has not been confirmed to work!
    # c = weights / # of symmetry elements
    c = weights / R.symmetry.shape[0]

    # f_hat = gcA2fourier(g,c,A); Bradley WIP

    # symmetrize f-hat
    # Has not been confirmed to work!
    # Ported from MATLAB from 'calcFourier.m' (Line 32)
    # Symmetrize crystal symmetry correction
    A_copy = A
    A_copy[0:] = 1;
    c_copy = numpy.ones((len(Psi.CS), 1))
    # f_hat_copy = multiply("gcA2fourier(component.SS, c, A)", c_copy, len(A_copy) - 1) # Bradley WIP

    # antipodal but leave commented out
    # Has not been confirmed to work!
    # Ported from MATLAB from 'calcFourier.m' (Line 47)
    # Correction for antipodal component
    """f_hat_copy = f_hat #Avoids overwrite
    for i in range(len(A) - 1):
        arr = np.arange(1, deg2Dim(i) - 1)
        index = deg2Dim(i) + arr
        f_hat_copy[index] = 0.5 * (numpy.reshape(f_hat_copy[index], 2*i+1, 2*i+1))
        + numpy.reshape(f_hat_copy[index], 2*i+1, 2*i+1))"""

    # return odf object
    # Must adhere to Stephen Egnaczyk's ODF data structure!
    # Create ODF Object from calculated values
    # ODF("f_hat", weights, L, "CS", "SS", "odfKernel")


# Has not been confirmed to work!
def multiply(f1, f2, lA):
    f1_copy = f1  # Avoids overwrite
    f2_copy = f2  # Avoids overwrite
    f = numpy.zeros((1, f1.size))
    for i in range(lA - 1):
        arr = numpy.arange(1, deg2Dim(i) - 1)
        index = deg2Dim(i) + arr;
        f[index] = numpy.reshape(f1_copy[index], 2 * i + 1, 2 * i + 1)
        *numpy.reshape(f2_copy[index], 2 * i + 1, 2 * i + 1)


# Simple helper function. Only used in this file.
def deg2Dim(l):
    return l * (2 * l - 1) * (2 * l + 1) / 3;