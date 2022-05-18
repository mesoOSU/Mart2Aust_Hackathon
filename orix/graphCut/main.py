import numpy as np
import matplotlib.pyplot as plt
from orix.graphCut.call_reconstruction import call_reconstruction
import scipy.sparse as sp

from orix import data, io, plot
from orix.io import load
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation, symmetry, Misorientation
from orix.vector import Vector3d

DATA_DIR = 'Mart2Aust/orix/graphCut/data/'
FILE_NAME = 'AF_001.hdf5'
FILE_PATH = DATA_DIR + FILE_NAME
# GRID_TYPE = 'SqrGrid' ### user defined as 'SqrGrid' or 'HexGrid'

options = {
    "options": "dictionary", 
    "GRID_TYPE": "SqrGrid", ### user defined as 'SqrGrid' or 'HexGrid'
    "MART_PHASE_ID": 2, # PHASE_ID = 2 ### number val or 'string' corresponding to mart phase, or name CHILD_PHASE_ID
    "max_recon_attempts": 10,
    'min_cut_size': 100
}

##### begin main

xmap = load(FILE_PATH)

options['NROWS'] = xmap.shape[0]
options['NCOLS_ODD'] = xmap.shape[1]

call_reconstruction(xmap, options)