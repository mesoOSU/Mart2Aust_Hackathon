# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:30:15 2022

@author: ashley
"""
#%%
from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion.rotation import Rotation

import tempfile

from diffpy.structure import Atom, Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import maxflow
from orix import data, io, plot
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation, symmetry, Misorientation
from orix.vector import Vector3d
plt.close('all')
import h5py
import time

#############  PLEASE READ  ##############
# the first 4 cells is just loading data from AF_001.hdf5 so you can look at phases
# you dont need to run these first 4 cells if you dont want to plot that so you can skip
# to cell 5 it you want (run this cell first before doing that)
##########################################

#%% get .ang file from EBSD.hdf5 file
file = r'C:\Users\ashle\Documents\GitHub\Mart2Aust_Hackathon\orix\graphCut\data\AF_001.hdf5'

f = h5py.File(file, 'r')
group = f['crystal_map']['data']
gKeys = list(group.keys())
euler1, euler2, euler3  = np.asarray(group['phi1']), np.asarray(group['Phi']), np.asarray(group['phi2']) # bunge convention
x, y, z = np.asarray(group['x']), np.asarray(group['y']), np.asarray(group['z'])
iq, ci, phase_id, ids, fit, is_in_data  = np.asarray(group['iq']), np.asarray(group['ci']), np.asarray(group['phase_id']), np.asarray(group['id']), np.asarray(group['fit']), np.asarray(group['is_in_data']) # i can't believe you looked all the way over here 
f.close()
#%%
# Create a Rotation object from Euler angles
euler_angles = np.column_stack((euler1, euler2, euler3))
rotations = Rotation.from_euler(euler_angles)

# Create a property dictionary
properties = dict(iq=iq, dp=ci)

# Create unit cells of the phases
structures = [
    Structure(
        title="austenite",
        atoms=[Atom("fe", [0] * 3)],
        lattice=Lattice(0.360, 0.360, 0.360, 90, 90, 90)
    ),
    Structure(
        title="ferrite",
        atoms=[Atom("fe", [0] * 3)],
        lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
    )
]
phase_list = PhaseList(
    names=["austenite", "ferrite"],
    point_groups=["432", "432"],
    structures=structures,
)

# Create a CrystalMap instance
xmap2 = CrystalMap(
    rotations=rotations,
    phase_id=phase_id,
    x=x,
    y=y,
    phase_list=phase_list,
    prop=properties,
)
xmap2.scan_unit = "um"
#%%

ckey_m3m = plot.IPFColorKeyTSL(xmap2.phases[1].point_group, direction=Vector3d.zvector())
rgb_au = ckey_m3m.orientation2color(xmap2["austenite"].orientations)
rgb_fe = ckey_m3m.orientation2color(xmap2["ferrite"].orientations)

#%% plot
xmap2["ferrite"].plot(rgb_fe)
xmap2["austenite"].plot(rgb_au)



###############################################################################################
#%% load data from graph cut in matlab
data_p = 'data/'
# load inplane weights (misorientations)
in_mat_data = loadmat(data_p+'ipw_w.mat') 
inplane = in_mat_data['ipw_w'].T 
# load out of plane weights (misorientation with random orientation)
out_mat_data = loadmat(data_p+'opw_w.mat')
outplane = out_mat_data['opw_w'].reshape(321,321)
# load node connections
nodes = loadmat(data_p + 'ipDict.mat')['connections'].T
#n create an array with node connections and corresponding weights 
for_net = np.hstack((nodes, inplane))

#%% rough graph cut
g = maxflow.GraphFloat()
nodeids = g.add_grid_nodes((321, 321))
# good_nodes[good_nodes<20000] = 0 # test blocking out section for graph cut

# assign weights from node connections in for_net
for i in range(len(for_net)):
    uu,vv, mwt = for_net[i, 0], for_net[i,1], for_net[i,2]
    g.add_edge(int(uu),int(vv),mwt,mwt)

# g.add_grid_tedges(nodeids, outplane, np.mean(outplane))   #test different weighting
# g.add_grid_tedges(nodeids, outplane, 1/outplane)          #test different weighting
g.add_grid_tedges(nodeids, outplane, 1-outplane)
g.maxflow()
sgm = g.get_grid_segments(nodeids)
img2 = np.int_(np.logical_not(sgm))
from matplotlib import pyplot as ppl
ppl.figure('rough cut')
ppl.imshow(img2)
ppl.show()

# %%set austenite grains nodes to 0 and pymaxflow will automagically ignore
# precision graph cut

# assign austenite grains nodes to 0 --> pymaxflow will now ignore these
p = maxflow.GraphFloat() # make a new graph (i dont know if this is actually necessary)
new_nodeids = p.add_grid_nodes((321, 321)) 
new_nodeids[sgm==True] = 0  # where old graph cut = true, set those nodes to 0 in nodes for this graph

#%%
start = time.time()
nzn = new_nodeids[new_nodeids!=0].flatten() # get non-zero node IDs

# make a new "for_net" variable without the zero node IDs
new_for_net = []
for ii in range(len(for_net)):
    test_u, test_v, test_wt = for_net[ii,:]
    if test_u in nzn and test_v in nzn:
        new_for_net.append((test_u, test_v, test_wt))
new_for_net = np.asarray(new_for_net)   

# set edges in new graph with old weights from "for_net" but with out the zero nodes
for j in range(len(new_for_net)):
    uuu, vvv, mmwt = new_for_net[j, 0], new_for_net[j,1], new_for_net[j,2]
    p.add_edge(int(uuu),int(vvv),mmwt,mmwt)
        
stop = time.time()
total_time = stop-start 
print('total time of for loop:', total_time) #just seeing how long this takes ~20 seconds

#%%
##### load different source weights for second cut
from orix.graphCut.temp_functions import yardley_variants
from call_reconstruction import get_op_weights_v2
from orix.io import load
xmap2 = load(file)
new_weights = get_op_weights_v2(xmap2)

# assign new source and sink weights 
p.add_grid_tedges(new_nodeids, new_weights, 1-new_weights)
p.maxflow()
sgm2 = p.get_grid_segments(new_nodeids)
img3 = np.int_(np.logical_not(sgm2))
ppl.figure('pruned2')
ppl.imshow(img3)
ppl.show()

## plot the difference between the first cut and the second
diff = img2 - img3
ppl.figure('diff')
ppl.imshow(diff)
ppl.show()







