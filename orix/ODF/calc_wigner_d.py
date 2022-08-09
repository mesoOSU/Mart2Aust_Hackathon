# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9 09:21:09 2022

@author: simon
"""

import numpy as np
from scipy.special import jacobi
from math import factorial, sqrt, sin, cos, radians

def precalc_reduced_terms(l):
    """ precalculate terms needed to find reduced matrix elements, d_lmn(PHI) - without needing PHI
        using Eqn 4.74 in Man's "Crystallographic Texture and Group Representations" to calculate d_lmn(PHI)
        precalculated terms remain the same for the same value of l

    Args:
        l (int): bandwidth (?? is that the term) for generating the Wigner D matrix

    Returns:
        terms: list of arrays where terms 0-4 correspond to terms in Eqn 4.74 in Man's "Crystallographic Texture and Group Representations"
                terms m and n keep track of m and n values corresponding to each index pair in Wigner D matrix

                each term array is 1x(2*l+1)*(2*l+1) vector containing terms of Wigner D calculation going left to right and top to bottom
    """

    dim = 2*l + 1 ### dimension of wigner d matrix

    term_0 = np.zeros((dim*dim)) ### term_0 == (-1)**(m-n)
    term_1 = np.zeros_like(term_0) ### term_1 == sqrt(factorial(l-m)*factorial(l+m)/(factorial(l-n)*factorial(l+n)))
    term_2 = np.zeros_like(term_0) ### term_2 == (m-n) for (sin(PHI/2))**(m-n)
    term_3 = np.zeros_like(term_0) ### term_3 == (m+n) for (cos(PHI/2))**(m+n)
    term_4 = [np.array([None], dtype=object)] ### term_4 == jacobi(l-m, m-n, m+n) for jacobi(l-m, m-n, m+n)(cos(PHI))
    term_m = np.zeros_like(term_0) ### keep track of current m value for D_LMN calc
    term_n = np.zeros_like(term_0) ### keep track of current n value for D_LMN calc
    term_l = np.zeros_like(term_0)

    count = -1

    for m in np.arange(l, -l-1, -1): ### iterate through indicies from l to -l
        # for n in np.arange(m, l): ### technically only need to calculate diagonal and upper triangle, lower triangle will be 0
        for n in np.arange(l, -l-1, -1):
            count += 1
            term_m[count] = m
            term_n[count] = n
            term_l[count] = l
            if (m-n >= 0) and (m+n >= 0): ### if these conditions fail, trying to take a factorial of a negative
                term_0[count] = (-1)**(m-n)
                term_1[count] = sqrt(factorial(l-m)*factorial(l+m)/(factorial(l-n)*factorial(l+n)))
                term_2[count] = (m-n)
                term_3[count] = (m+n)
                term_4.append(jacobi(l-m, m-n, m+n)) ### jacobi returns poly1d array, needed to append one at a time instead of preallocate
            else:
                term_4.append(0)

    term_4 = term_4[1:] ### jacobi array initialized with a None object, remove this object

    return [term_0, term_1, term_2, term_3, term_4, term_m, term_n, term_l] ### returns a list of term arrays

def stack_all_precalc_terms(l):
    """find all sets of terms for each bandwidth [1...l]
        vectorizes and calculates each set and stacks all terms into 1d vectors of terms


    Args:
        l (int): bandwidth for generating the Wigner D matrix

    Returns:
        terms: list of arrays of 1x(2*l+1)^2 for all l terms, with each bandwidth set being appended on the end of the preceeding set
    """
    l_vector = np.arange(1,l+1)

    vect_precalc = np.vectorize(precalc_reduced_terms)
    all_terms = vect_precalc(l_vector)

    terms = all_terms[0]
    for i,_ in enumerate(all_terms):
        terms[0] = np.append(terms[0], all_terms[i][0])
        terms[1] = np.append(terms[1], all_terms[i][1])
        terms[2] = np.append(terms[2], all_terms[i][2])
        terms[3] = np.append(terms[3], all_terms[i][3])
        terms[4] = np.append(terms[4], all_terms[i][4])
        terms[5] = np.append(terms[5], all_terms[i][5])
        terms[6] = np.append(terms[6], all_terms[i][6])
        terms[7] = np.append(terms[7], all_terms[i][7])

    return terms

def calc_Wigner_D(phi1, PHI, phi2, terms):
    """given a rotation, R(phi1, PHI, phi2), calculate Wigner D active rotation matrix, D_LMN(R), using a precalculated terms array for a given bandwidth
        using Eqn 4.18 in Man's "Crystallographic Texture and Group Representations" to calculate D_LMN(R(phi1,PHI,phi2))

        ###NOTE probably will need to change input from three separate angles to a rotation object when these functions absorbed into ODF class

        instead of calculating d_lmn in separate function, just do it here

    Args:
        phi1 (float): angle in radians 
        PHI (float): angle in radians
        phi2 (float): angle in radians
        terms (list of arrays): _description_

    Returns:
        D_LMN: return 1x(2*l+1)^2 for all values of l vector array of all D_LMN values
    """

    d_lmn = np.asarray([terms[0][i]*terms[1][i]*(sin(PHI/2)**terms[2][i])*(cos(PHI/2)**terms[3][i])*terms[4][i](cos(PHI)) if terms[4][i] !=0 else 0 for (i,_) in enumerate(terms[0])])

    D_LMN = np.asarray([np.exp(-1j*terms[5][i]*phi1)*d_lmn[i]*np.exp(-1j*terms[6][i]*phi2) if d_lmn[i] !=0 else 0 for (i,_) in enumerate(d_lmn)])

    return D_LMN

# def calc_reduced_d(PHI, terms):
#     """
#         given the precalculated terms and the theta/PHI angle, calculate reduced matrix elements, d_lmn(PHI)

#     Args:
#         PHI (float): angle in radians for theta/phi/beta angle
#         terms (list of arrays): _description_

#     Returns:
#         d_lmn: return 1x(2*l+1)^2 vector array of all d_lmn values
#     """

#     d_lmn = np.asarray([terms[0][i]*terms[1][i]*(sin(PHI/2)**terms[2][i])*(cos(PHI/2)**terms[3][i])*terms[4][i](cos(PHI)) if terms[4][i] !=0 else 0 for (i,_) in enumerate(terms[0])])

#     return d_lmn

# def calc_Wigner_D_alternate(phi1, PHI, phi2, terms):
#     """given a rotation, R(phi1, PHI, phi2), calculate Wigner D active rotation matrix, D_LMN(R), using a precalculated terms array for a given bandwidth
#         using Eqn 4.18 in Man's "Crystallographic Texture and Group Representations" to calculate D_LMN(R(phi1,PHI,phi2))

#         ###NOTE probably will need to change input from three separate angles to a rotation object when these functions absorbed into ODF class

#         reduced order d_lmn in separate function, not really needed to be separate
#     Args:
#         phi1 (float): angle in radians 
#         PHI (float): angle in radians
#         phi2 (float): angle in radians
#         terms (list of arrays): _description_

#     Returns:
#         D_LMN: return 1x(2*l+1)^2 vector array of all D_LMN values
#     """

#     d_lmn = calc_reduced_d(PHI, terms)

#     D_LMN = np.asarray([np.exp(-1j*terms[5][i]*phi1)*d_lmn[i]*np.exp(-1j*terms[6][i]*phi2) if d_lmn[i] !=0 else 0 for (i,_) in enumerate(d_lmn)])

#     return D_LMN