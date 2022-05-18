import operator

import numpy
from numpy import sqrt
from numpy import empty, zeros
from numpy.linalg import norm
from math import acos, cos
from math import pi

import numpy as np

from scipy.spatial.transform import Rotation as R


def vecarrayconvert(a):  # FIXME: Not necessary? Can just use asanyarray?
    ''' convert any reasonable datatype into an n x 3 vector array'''
    import numpy as np

    # a = np.asanyarray(a)

    # a = np.squeeze( np.float64( list( (a,) )))
    # a = np.atleast_2d(np.asarray(a))

    # Make the arrays indexable if they are actually scalars
    # if np.size(a) == 1:
    #    a = np.array([a])

    a = np.atleast_1d(np.squeeze(np.asanyarray(a)))

    return a


def vecarraynorm(a):
    nrm = []

    ''' norm of an array of vectors '''
    import numpy as np
    ax = vecarrayconvert(a[0])
    ay = vecarrayconvert(a[1])
    az = vecarrayconvert(a[2])

    if np.shape(ax) == np.shape(ay) == np.shape(az):
        nrm = (ax * ax + ay * ay + az * az) ** 0.5
    else:
        print("vecarraynorm error: check that the lengths of arguments are equal.")

    return nrm


def uniquerows(aa):
    ''' returns the number of unique rows in an array.

    Parameters
    ----------
    aa : numpy array

    Returns
    -------
    cc : numpy array
        Rows in aa without repetitions
    ia : numpy Bool array
        Indexing array such that cc = aa[ia]
    ic : numpy index array
        Indexing array such that aa = cc[ic]

    Notes
    -----
    Mimics behavior of matlab function 'unique' with optional parameter 'rows'.
    Algorithm modified from a stack overflow posting [1]_.

    References
    ---------
    .. [1] http://stackoverflow.com/questions/8560440/, Accessed 2012-09-10
    '''
    import numpy as np

    ind = np.lexsort(np.fliplr(aa).T)  # indices for aa sorted by rows
    rev = np.argsort(ind)  # reverse of the sorting indices
    ab = aa[ind]  # ab is sorted version of aa
    dd = np.diff(ab, axis=0)  # get differences between the rows
    ui = np.ones(np.shape(ab)[0], 'bool')  # preallocate boolean array
    ui[1:] = (dd != 0).any(axis=1)  # if difference is zero, row is not unique

    ia = ui[rev]  # positions of unique rows in aa (original, unsorted array)
    cc = aa[ia]  # unique rows in aa in original order

    loc = np.cumsum(np.uint64(ui)) - 1  # cumulative sum := locs of repeats in ab

    # - rev[ia] gives us the indices of the unique rows in aa
    # - argsort(rev[ia]) gives us the indices of the corresponding rows in cc
    # - argsort(rev[ia])[loc] gives us the indexing relationship for ab from cc
    # - np.argsort(rev[ia])[loc][rev] give indexing reln in original order
    ic = np.argsort(rev[ia])[loc][rev]

    return cc, ia, ic


def sigdec(a, n=1):
    '''
    Rounds the elements of a to n decimals.
    A slight modification of Peter J. Acklam's Matlab function FIXDIG
    '''
    import numpy as np
    a = vecarrayconvert(a)
    n = vecarrayconvert(n)
    f = np.array(10. ** n)
    s = np.sign(a)

    return s * np.around(np.absolute(a) * f) / f


def namedOR(name):
    """ Returns ksi values for named orientation relationships

    Parameters
    ----------
    name : {'ks', 'nw', 'bain'}
        Orientation relationship name

    Returns
    -------
    ksi_values : 1x3 numpy array of floats
        ksi values in degrees

    Notes
    -----
    TODO: Add plane parallel Greninger-Troiano, Kelly, etc.
    TODO: Allow 'Kurdjumov-Sachs' as well as 'ks', etc.
    """
    import numpy as np
    if isinstance(name, str):
        if name.lower() == 'ks':
            s6 = np.sqrt(6.0)
            s3 = np.sqrt(3.0)
            ksi1 = np.arccos((s6 + 1.0) / (2.0 * s3))
            ksi2 = np.arccos((s6 + 18.0) / (12.0 * s3))
            ksi3 = np.arccos((s6 + 12.0) / (6.0 * s6))
            ksi = np.array([ksi1, ksi2, ksi3])
            del s6, s3, ksi1, ksi2, ksi3
        elif name.lower() == 'nw':
            s6 = np.sqrt(6)
            s2 = np.sqrt(2)
            ksi0 = np.arccos((s2 + 1.0) / s6)
            ksi = np.array([0.0, ksi0, ksi0])
        elif name.lower() == 'bain':
            ksi = np.array([0.0, 0.0, 0.0])
        else:
            print('namedOR: Unrecognized named OR')
    else:
        print('namedOR requires a string input. Returning Bain.')
        ksi = np.array([0.0, 0.0, 0.0])

    return ksi * 180.0 / np.pi


def yardley_variants(ksi_values):
    '''
    YardleyVariants returns the matrices corresponding to the variants
                    produced from the provided orientation relationship,
                    specified in Kurjumov-Sachs angles.

    In the case of the Kurdjumov-Sachs or Nishiyama-Wasserman orientation
    relationships, 'KS' or 'NW' can be passed to the function as OR.

    If `OR` is numeric, this function assumes that `OR`'s first three values
    are in radians.

    -------------------------------------------------------------------------
    2010-11-25 | Victoria Yardley (victoria.yardley[at]rub.de)
                Eric Payton (payton.28[at]osu.edu)
                Ruhr-Universitaet Bochum
    %%%         Matlab function written by EP based on VY's             %%% %
    %%%           spreadsheet for calculation of variants               %%% %
    -------------------------------------------------------------------------
    This program is provided without any guarantee of correctness.
    If you modify it and/or improve it, please kindly share with me your new
    and improved version to the email address above. Thanks!

    Dependencies:
    --------------
    import operator
    from numpy import sqrt
    from numpy import empty, zeros
    from numpy.linalg import norm
    from math import acos, cos
    from math import pi
    '''
    """ Generate variants from Kurdjumov-Sachs angles

    Returns matrices of an orientation relationship specified in Kurjumov-Sachs
    angles.

    Parameters
    ----------
    ksi_values : length 3 iterable OR {'KS', 'NW', 'Bain'}

    Returns
    -------
    vv : rmat object
        rotation matrices corresponding to variants

    """

    if isinstance(ksi_values, str):
        ksi = namedOR(ksi_values)
    else:
        ksi = ksi_values

    ksi = [ksi[i] * np.pi / 180 for i in range(3)]

    # convert ksi radians to rotation matrices

    mb = np.zeros([2, 9])

    mb[0, 0] = np.cos(ksi[0])
    mb[0, 4] = np.cos(ksi[1])
    mb[0, 8] = np.cos(ksi[2])

    costh = 0.5 * (np.sum(np.cos(ksi)) - 1.0)  # sum(cos(ksi)) is the matrix trace
    mosth = 1.0 - costh
    sinth = np.sqrt(1.0 - costh ** 2.0)

    r1 = np.sqrt((mb[0, 0] - costh) / mosth)
    r2 = np.sqrt((mb[0, 4] - costh) / mosth)
    r3 = np.sqrt((mb[0, 8] - costh) / mosth)
    del costh

    r1r2 = r1 * r2 * mosth
    r1r3 = r1 * r3 * mosth
    r2r3 = r2 * r3 * mosth
    r3st = r3 * sinth
    r2st = r2 * sinth
    r1st = r1 * sinth
    del r1, r2, r3, mosth, sinth

    mb[0, 5] = r2r3 - r1st
    mb[0, 7] = r2r3 + r1st
    mb[1, :] = mb[0, :]

    mb[0, 1] = -r1r2 + r3st
    mb[0, 2] = -r1r3 - r2st
    mb[0, 3] = -r1r2 - r3st
    mb[0, 6] = -r1r3 + r2st
    del r1r2, r1r3, r2r3, r3st, r2st, r1st

    mb[1, 1] = -mb[0, 1]
    mb[1, 2] = -mb[0, 2]
    mb[1, 3] = -mb[0, 3]
    mb[1, 6] = -mb[0, 6]
    # mb[0] is the 'positive' solution; mb[1] is the 'negative' solution

    # create Bain correspondence matrices
    bb = np.zeros([12, 9])
    bb[0, :] = [1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    bb[1, :] = [0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    bb[2, :] = [-1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    bb[3, :] = [0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0]
    bb[4, :] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0]
    bb[5, :] = [1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
    bb[6, :] = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    bb[7, :] = [-1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    bb[8, :] = [0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0]
    bb[9, :] = [1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    bb[10, :] = [0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0]
    bb[11, :] = [-1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0]

    # normalize correspondence matrices
    # This is a mess, yell at Austin if you need help
    bb33 = [bb[i].reshape(3, 3) for i in np.arange(12)]

    bb = [bb33[i] / np.sum(bb33[i] ** 2, axis=0) ** 0.5 for i in np.arange(12)]

    mb_new = np.zeros([2, 3, 3])

    mb_new[0] = mb[0].reshape(3, 3)
    mb_new[1] = mb[1].reshape(3, 3)

    mb = mb_new * 1

    # produce variants
    vv = np.zeros([24, 3, 3])

    j = 0
    for i in range(0, len(bb)):

        temp1 = np.dot(mb[0], bb[i])
        vv[j] = temp1

        j = j + 1

        temp2 = np.dot(mb[1], bb[i])
        vv[j] = temp2

        j = j + 1

    vv = vv.reshape(24, 9)

    # reduce redundancies, if they exist (as they do, for example, in NW)
    vv, ia, ic = uniquerows(sigdec(vv, 7))

    print(vv.shape)

    num = int(vv.size / vv[0].size)

    vv = vv.reshape(num,3,3)

    return vv


def main():
    ksiKS = [5.26, 10.30, 10.53]
    ksiNW = [0, 9.74, 9.74]

    yvKS = yardley_variants(ksiKS)
    tvNW = yardley_variants(ksiNW)

main()
