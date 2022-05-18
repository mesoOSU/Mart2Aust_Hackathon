"""
Created on Fri May 13, 2022
@author: StevenEgnaczyk
"""

import math
import numpy
import scipy.special as sc


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
    def __init__(self, f_hat, weights, bandwidth, CS, SS, odfKernel):
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

    # Calculate Psi values
    kappa = math.floor((0.5 * math.log(0.5)) / math.log(math.cos(halfwidth / 2)))
    C = round(sc.beta(1.5, 0.5) / sc.beta(1.5, kappa + 0.5))
    A = []
    bandwidth = 90

    return Psi(kappa, C, A, bandwidth)


def calc_odf(orientations, odfKernel):
    """
    Calculates the ODF given the orientations and kernel.

    :param orientations: (List of orientations)
        Note: Add what this is

    :param odfKernel: (Psi Object)
        Note: Add what this is

    :return:
        An ODF object
    """

    L = math.floor(odfKernel.bandwidth)

    # Create and cut the A Array
    odfKernel.A = numpy.ones((1, L + 1))
    odfKernel.A = odfKernel.A[0, :]

    # Calculate Chebyshev coefficients
    odfKernel.A[1] = odfKernel.kappa / (odfKernel.kappa + 2.0)

    for j in range(1, L):
        odfKernel.A[j + 1] = ((odfKernel.kappa - j + 1) * odfKernel.A[j - 1] - (2 * j + 1) * odfKernel.A[j]) \
                             / (odfKernel.kappa + j + 2)

    for i in range(0, L):
        odfKernel.A[i] = (2 * i + 1) * odfKernel.A[i]

    # Cut off Chebyshev coefficients when the values are small
    odfKernel.A = cut(odfKernel.A)

    # Return the ODF object
    return ODF(0, 0, 0, 0, 0, odfKernel)


def cut(A):
    """
    Cuts the Chebyshev coefficients off when they get too small

    :param A:
        List of Chebyshev coefficients

    :return:
        A smaller list of Chebyshev coefficients
    """

    # Epsilon value denoting the cutoff point for Chebyshev values
    epsilon = 0.0100
    returnA = []

    # Loop through the array and add the large values
    for i in range(-1, len(A) - 1):
        if A[i] > epsilon:
            returnA.append(round(A[i], 4))

    # Return the smaller list of Chebyshev coefficients
    return returnA


def main():

    odfKernel = generate_kernel(0.1745)
    odfObject = calc_odf([], odfKernel)
    print(odfObject.odfKernel.A)

    odfKernel2 = generate_kernel(0.34634)
    odfObject2 = calc_odf([], odfKernel2)
    print(odfObject2.odfKernel.A)


main()
