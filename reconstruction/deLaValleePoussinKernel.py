"""
Created on Fri May 13, 2022
@author: Steven Egnaczyk, Jonathan Cappola
"""

import math
import numpy
import numpy as np
import scipy.special as sc

import orix.quaternion
from orix.quaternion import Misorientation


class Psi:
    """
    Psi Object Class.
    """

    # ============================ #
    #          Constructor         #
    # ============================ #
    def __init__(self, kappa, c, a, bandwidth):
        """
        Generates a Psi object, given Kappa, C, A, and bandwidth values

        :param kappa: (Integer)
            # Note: Add what Kappa is

        :param c: (Array)
            # Note: Add what Kappa is

        :param a: (Array)
            # Note: Add what a is

        :param bandwidth: (Float Value)
            # Note: Add what bandwidth is
        """

        self.kappa = kappa
        self.C = c
        self.A = a
        self.bandwidth = bandwidth


class ODF:
    """
    ODF Object Class.
    """

    # ============================ #
    #          Constructor         #
    # ============================ #
    def __init__(self, f_hat, weights, bandwidth, CS, SS, odfKernel, center):
        """
        :param f_hat:
            # To-Do: Add what this is

        :param weights:
            # To-Do: Add what this is

        :param bandwidth:
            # To-Do: Add what this is

        :param CS:
            # To-Do: Add what this is

        :param SS:
            # To-Do: Add what this is

        :param odfKernel:
            # To-Do: Add what this is
        """

        # Set variables
        self.f_hat = f_hat
        self.weights = weights
        self.bandwidth = bandwidth
        self.CS = CS
        self.SS = SS
        self.odfKernel = odfKernel
        self.center = center


# ============================ #
#          Methods             #
# ============================ #

def generate_kernel(halfwidth):
    """
    Creates a Kernel Object given the half-width.

    :param halfwidth: (Float)
        Add what this is

    :return:
        A Psi Object, the Kernel
    """

    # Calculate kappa and the pre-cos constant
    kappa = 0.5 * math.log(0.5) / math.log(math.cos(halfwidth / 2))
    C = sc.beta(1.5, 0.5) / sc.beta(1.5, kappa + 0.5)

    # Extract the bandwidth as an int
    L = math.floor(kappa)

    # Compute Chebyshev coefficients
    A = numpy.ones((L + 1,))  # 0 to L
    A[1] = kappa / (kappa + 2)  # Overwrite second value
    for j in range(1, L):
        A[j + 1] = ((kappa - j + 1) * A[j - 1] - (2 * j + 1) * A[j]) \
                   / (kappa + j + 2)

    for i in range(0, L):
        A[i] = (2 * i + 1) * A[i]

    A = cut(A)

    # Set the bandwidth per length of A
    bandwidth = A.size - 1

    return Psi(kappa, C, A, bandwidth)


def cut(A):
    """
    Cuts the Chebyshev coefficients off when they get too small

    :param A:
        List of Chebyshev coefficients

    :return:
        A smaller list of Chebyshev coefficients
    """

    # Epsilon value denoting the cutoff point for Chebyshev values
    epsilon = 0.0100 / 150

    returnA = A / numpy.arange(1, A.size + 1) ** 2
    ind = numpy.argwhere(returnA[1:] <= numpy.maximum(numpy.amin(
        numpy.append(returnA[1:], 10 * epsilon)), epsilon))[0]

    returnA = A[:numpy.minimum(int(ind + 1), A.size - 1) + 1]  # Added a +1 here!
    
    # Return the smaller list of Chebyshev coefficients
    return returnA


def eval_kernel_odf(odf, g):

    # Compute the orix misorientation and the angle between the quats
    w = g.angle_with_outer(odf.center)

    # Some math preliminaries
    base = math.cos(w / 2.0)
    power = 2.0 * odf.odfKernel.kappa
    
    # Evaluate the kernel value directly
    v = odf.odfKernel.C * np.power(base, power)
    
    # Divide the result by the CS and SS multiplicities to match MTEX expectation
    v = v / odf.CS.laue_proper_subgroup.size / odf.SS.laue_proper_subgroup.size

    return v


def main():
    """ Test program to show how to generate and evalutate a unimodalODF
    similar to how MTEX resolves it. This follows the equations of MTEX and
    those outlined in:
        The de le Vallee Poussin Standard Orientation Density Function
        H. Schaeben (1999)
        https://doi.org/10.1155/TSM.33.365
    """
    
    # Assign the halfwidth value in radians that we want to use
    hw = numpy.radians(2.50) # Degree value as input, radian as output
    
    # Generate the unimodal kernel using the halfwidth
    odfKernel = generate_kernel(hw)
    
    # Construct the crystal and specimen symmetries using orix
    SS = orix.quaternion.Symmetry([1, 0, 0, 0]) # Triclinic '-1'
    CS = orix.quaternion.symmetry.get_point_group(225) # Cubic 'm-3m'

    # Assemble a "center" orientation for the unimodal ODF using orix
    ori_center = numpy.radians([107.028, 43.8449, 260.180]) # Input degree, output radian
    center = orix.quaternion.Orientation.from_euler(ori_center, CS, direction='MTEX')

    # Add the kernel and the center orientation to the odf object - This may need to change later!!!
    odf = ODF([], [], odfKernel.bandwidth, CS, SS, odfKernel, center)

    # Assemble a "g" orientation used to evaluate the unimodal ODF using orix
    ori_g = numpy.radians([106.0, 42.0, 259.0]) # Input degree, output radian
    g = orix.quaternion.Orientation.from_euler(ori_g, CS, direction='MTEX')

    # Evaluate the ODF at "g"
    v = eval_kernel_odf(odf, g)
    
    return v, odfKernel

v, kernel = main()
