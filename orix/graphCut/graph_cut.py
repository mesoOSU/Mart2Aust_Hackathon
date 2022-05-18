from mimetypes import init
import struct
import numpy as np
import matplotlib.pyplot as plt
import maxflow
import scipy.sparse as sp
import networkx as nx

from orix import data, io, plot
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation, symmetry, Misorientation
from orix.vector import Vector3d

from graphCutUtils import init_grid_graph, init_user_defined_graph

###TODO make graph cut object

class graph_cut(object):
    def __init__(self, nrows, ncols_odd, grid, updated_nxgraph=None, op_weights_arr=None):
        self.g = maxflow.GraphFloat()
        self.nrows = nrows
        self.ncols_odd = ncols_odd
        self.grid = grid

        if updated_nxgraph == None and op_weights_arr == None:
            init_grid_graph(self)
        else:
            init_user_defined_graph(self)

        self.nodeids
        self.nxGraph

        self.params

        # self.ip_weight
        # self.op_weight