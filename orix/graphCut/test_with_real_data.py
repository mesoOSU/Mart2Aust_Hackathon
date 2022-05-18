# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:30:15 2022

@author: ashley
"""

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
# f.close()
#%% get .ang file from EBSD.hdf5 file

file = r"C:\Users\ashle\Downloads\AF_001.hdf5"

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

#%% load data from graph cut in matlab
data_p = r'C:\Users\ashle\Documents\GitHub\Mart2Aust_Hackathon\orix\graphCut\data\\'
in_mat_data = loadmat(data_p+'ipw_w.mat')
inplane = in_mat_data['ipw_w'].T
out_mat_data = loadmat(data_p+'opw_w.mat')
outplane = out_mat_data['opw_w'].reshape(321,321)
nodes = loadmat(data_p + 'ipDict.mat')['connections'].T
for_net = np.hstack((nodes, inplane))
#%% graph cut
g = maxflow.GraphFloat()
nodeids = g.add_grid_nodes((321, 321))

for i in range(len(for_net)):
    uu,vv, mwt = for_net[i, 0], for_net[i,1], for_net[i,2]
    g.add_edge(int(uu),int(vv),mwt,mwt)

# g.add_grid_tedges(nodeids, outplane, np.mean(outplane))
g.add_grid_tedges(nodeids, outplane, 1/outplane)
# g.add_grid_tedges(nodeids, outplane, 1-outplane)
g.maxflow()
sgm = g.get_grid_segments(nodeids)
img2 = np.int_(np.logical_not(sgm))
from matplotlib import pyplot as ppl
ppl.figure()
ppl.imshow(img2)
ppl.show()




