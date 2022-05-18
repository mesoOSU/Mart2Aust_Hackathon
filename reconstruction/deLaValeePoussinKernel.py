"""
Created on Fri May 13, 2022
@author: StevenEgnaczyk
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

    arr1 = [g.a[0], g.b[0], g.c[0], g.d[0]]
    arr2 = [odf.center.a[0], odf.center.b[0], odf.center.c[0], odf.center.d[0]]

    print(arr1)
    print(arr2)

    w = Misorientation(np.array([arr1, arr2]))

    w = w.get_distance_matrix()

    print(math.degrees(numpy.amax(w)))

    angle = 2 * math.acos(numpy.amax(w))

    print(angle)

    base = math.cos(angle / 2)

    power = odf.odfKernel.kappa

    print(base)

    print(power)

    print(odf.odfKernel.C)

    v = odf.odfKernel.C * np.power(base, power)

    print(v)

    return v


def main():
    odfKernel = generate_kernel(0.04360)

    SS = orix.quaternion.Symmetry([1, 0, 0, 0])
    CS = orix.quaternion.symmetry.get_point_group(225)

    rad1 = math.radians(107.028)
    rad2 = math.radians(43.8449)
    rad3 = math.radians(260.18)

    data = [rad1, rad2, rad3]
    center = orix.quaternion.Orientation.from_euler(data, CS)


    odf = ODF([], [], [], [], [], odfKernel, center)

    rad4 = math.radians(106)
    rad5 = math.radians(42)
    rad6 = math.radians(259)

    ori = [rad4, rad5, rad6]

    oriPrime = orix.quaternion.Orientation.from_euler(ori, CS)

    v = eval_kernel_odf(odf, oriPrime)

    return odfKernel


kernel = main()
