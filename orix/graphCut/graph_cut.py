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

###TODO make graph cut object

class graph_cut(object):
    def __init__(self, nrows, ncols_odd, grid, ip_data=None, op_weights=None):
        """_summary_

        creates a pymaxflow graphFloat object
        either creates a regular gridded network for extracting connectivity
        or creates a user-defined network for actual graph cutting

        Args:
            nrows (_type_): number of rows in ebsd data
            ncols_odd (_type_): number of odd columns in ebsd data, odd=even in square, but for hex, 
                                odd is longer row for generating connectivity matrix
            grid (_type_): descriptor for square or hex ebsd grid data
            ip_data (_type_, optional): in plane connections and weights used to create user defined graph. Defaults to None.
            op_weights (_type_, optional):out of plane weights used to create user defined graph. Defaults to None.
        """
        self.g = maxflow.GraphFloat()
        self.nrows = nrows
        self.ncols_odd = ncols_odd
        self.grid = grid

        # if ip_data == None and op_weights == None:
        #     self.init_grid_graph()
        # else:
        #     self.init_user_defined_graph(ip_data, op_weights)

        try:
            self.init_user_defined_graph(ip_data, op_weights)
        except:
            self.init_grid_graph()

        self.nodeids
        self.nxGraph
        # self.ip_weight
        # self.op_weight

    def init_grid_graph(self):
        """_summary_
            init grid-based graph to get connectivity
        """

        ip_weight = 0.01 ### max > ip_weight > min of data as a place to start

        # g = maxflow.GraphFloat()
        if self.grid == 'SqrGrid':
            self.nodeids = self.g.add_grid_nodes((self.nrows,self.ncols_odd))
            structure = maxflow.vonNeumann_structure(ndim=2, directed=True) ### square grid structure
        elif self.grid == 'HexGrid':
            self.nodeids = self.g.add_grid_nodes((self.nrows, self.ncols_odd))
            structure = np.array([[0, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]]) ### struct for hex grid
        else:
            print('define ebsd grid type as "SqrGrid" or "HexGrid"')
            
        self.g.add_grid_edges(self.nodeids, ip_weight, structure=structure)
        self.g.add_grid_tedges(self.nodeids, 1, 0)

        self.nxGraph = self.g.get_nx_graph()

    def init_user_defined_graph(self, ip_data, op_weights, op_weighting_type='inverse'):
        """_summary_
            init graph with user-defined structure - predetermined with grid, but edges need to be made one by one
            in order to create them with calculated ip weights

        Args:
            ip_data (_type_): _description_
            op_weights (_type_): _description_
            op_weighting_type (str, optional): user selected op weighting depending on style of cut. Defaults to 'inverse'.
        """

        op_weights_arr = np.reshape(op_weights, (self.nrows, self.ncols_odd))

        self.nodeids = self.g.add_grid_nodes((self.nrows, self.ncols_odd)) 

        for i in range(len(ip_data)):
            u,v, wt = ip_data[i, 0], ip_data[i,1], ip_data[i,2]
            self.g.add_edge(int(u),int(v),wt,wt)

        if op_weighting_type == 'mean':
            self.g.add_grid_tedges(self.nodeids, op_weights_arr, np.mean(op_weights_arr))
        elif op_weighting_type == 'inverse':
            self.g.add_grid_tedges(self.nodeids, op_weights_arr, 1/op_weights_arr)
        elif op_weighting_type == 'minus':
            self.g.add_grid_tedges(self.nodeids, op_weights_arr, 1 - op_weights_arr)

        self.nxGraph = self.g.get_nx_graph()

    def get_connectivity(self):
        """
        Returns
        -------
        inplane_connectivity : Array of int32
            returns the connectivity of inplane nodes of a network.
        """
        adj_arr = nx.to_scipy_sparse_array(self.g.get_nx_graph())
        sparseUpperArr = sp.triu(adj_arr)
        u,v,wt = sp.find(sparseUpperArr)
        full_connectivity = np.asanyarray([u,v])
        
        sink = np.amax(full_connectivity.ravel())
        source = sink-1
        source_edges = np.any(full_connectivity==source,axis=0)
        sink_edges = np.any(full_connectivity==sink, axis=0)
        out_plane_loc = np.any(np.vstack([source_edges, sink_edges]), axis=0)
        inplane_connectivity = full_connectivity[:, ~out_plane_loc] # adjacency matrix without source and sink connections
        
        return inplane_connectivity.T

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
