from diffpy.structure import Atom, Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np
import os

from orix import plot
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io import load, save
from orix.quaternion import Orientation, Rotation, symmetry
from orix.vector import Vector3d


plt.rcParams.update({"figure.figsize": (7, 7), "font.size": 15})



tempdir = '/Pyrepo/Hackathon/maxflow_for_matsci/Data/'
fname = "steel_ebsd.ang"

target = os.path.join(tempdir, fname)

xmap = load(target)
xmap.plot()  # Dot product values added to the alpha (RGBA) channel
print(xmap)
