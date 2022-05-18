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

def init_grid_graph(self):
### init grid-based graph to get connectivity

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

    return self.nxGraph

def init_user_defined_graph(self):

    self.nodeids = mister.add_grid_nodes((self.nrows, NCOLS_ODD)) 

    for i in range(len(updated_nxgraph)):
        uu,vv, mwt = updated_nxgraph[i, 0], updated_nxgraph[i,1], updated_nxgraph[i,2]
        if (uu>=sink or vv>=sink)==False:
            mister.add_edge(int(uu),int(vv),mwt,mwt)

    mister.add_grid_tedges(n_ids, op_weights_arr, 1/op_weights_arr)
    mister.maxflow()
    sgm = mister.get_grid_segments(n_ids)
    img3 = np.int_(np.logical_not(sgm))

    print('asdf')

def get_connectivity(self):
    """
    Returns
    -------
    inplane_connectivity : Array of int32
        returns the connectivity of inplane nodes of a network.
    """
    adj_arr = nx.to_scipy_sparse_array(self.get_nx_graph())
    sparseUpperArr = sp.triu(adj_arr)
    u,v,wt = sp.find(sparseUpperArr)
    full_connectivity = np.asanyarray([u,v])
    
    sink = np.amax(full_connectivity.ravel())
    source = sink-1
    source_edges = np.any(full_connectivity==source,axis=0)
    sink_edges = np.any(full_connectivity==sink, axis=0)
    out_plane_loc = np.any(np.vstack([source_edges, sink_edges]), axis=0)
    inplane_connectivity = full_connectivity[:, ~out_plane_loc] # adjacency matrix without source and sink connections
    return inplane_connectivity


# inputs: xmap, (inplane)connect
# outputs: misorientation for each node, random orientation (inplane and out of plane weights)


def cut_graph(self):
    
    self.g.maxflow()

    sgm = self.g.get_grid_segments(self.nodeids)
    print(np.sum(sgm))
    img2 = np.int_(np.logical_not(sgm))

    plt.imshow(img2)
    if self.grid == 'HexGrid':
        axes=plt.gca()
        axes.set_aspect(0.5)
    plt.show()

    return img2

# def cut_graph2(self):
#     #%% populate network through pymaxflow
#     defined_graph = maxflow.GraphFloat()
#     n_ids = defined_graph.add_grid_nodes((self.nrows, self.ncols_odd)) 

#     for i in range(len(updated_nxgraph)):
#         uu,vv, mwt = updated_nxgraph[i, 0], updated_nxgraph[i,1], updated_nxgraph[i,2]
#         if (uu>=sink or vv>=sink)==False:
#             defined_graph.add_edge(int(uu),int(vv),mwt,mwt)

#     defined_graph.add_grid_tedges(n_ids, op_weights, 1-op_weights)
#     defined_graph.maxflow()
#     sgm = defined_graph.get_grid_segments(n_ids)
#     img3 = np.int_(np.logical_not(sgm))
#     plt.figure("first")
#     plt.imshow(img3)
#     plt.show()

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