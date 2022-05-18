# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:41:19 2022

@author: ashley

duplicates the bottom diagonal from the top diagonal of a sparse matrix.
Can return the coordinates and values or just the matrix
"""
import scipy.sparse as sp
import numpy as np

def full_from_diag(diag_sparse, output_type='coordinates'):
    '''
    
    Parameters
    ----------
    diag_sparse : sparse._coo.coo_matrix
        diagonal sparse matrix that you want to duplicate for the bottom half
    output_type : compressed sparse column matrix or numpy array, optional
        Option to return the full matrix using 'sparse_matrix' or a compressed sparse column matrix or the rows, columns and values of the matrix.
        The default is 'coordinates'.
        
    Returns
    -------
    see output_type

    '''
    
    upper_half = sp.csr_matrix(diag_sparse) # original top diag
    lower_half = sp.csr_matrix(diag_sparse).T #duplicate for bottom diag
    full = lower_half + upper_half # throw it all together
    if output_type == 'sparse_matrix':
        return full
    
    if output_type == 'coordinates':
        coord_full_matrix = full.tocoo()
        new_coord_vals = np.asarray((coord_full_matrix.row, coord_full_matrix.col, coord_full_matrix.data)).T
        return new_coord_vals
    else:
        print("select output type, either 'coordinates' or 'sparse_matrix'")


