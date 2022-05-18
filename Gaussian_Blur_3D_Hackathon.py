# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:58:12 2022

@author: Charles Xu
@author: Bradley Rucker
"""

import numpy as np
import math

# Generation of the grid
def generate_grid():
    x = np.arange(-1, 1, 0.05)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-1, 1, 0.05)
    z = np.arange(-1, 1, 0.05)

    return x.reshape(x.shape[0],1), y.reshape(y.shape[0],1), z.reshape(z.shape[0],1)


# Generation of the blur in each direction
def gaussian_blur(xx, yy, zz, sigma = 1):
    fx = (1 / (2 * math.pi * sigma)) * np.exp((-xx ** 2) / (2 * sigma ** 2))
    fy = (1 / (2 * math.pi * sigma)) * np.exp((-yy ** 2) / (2 * sigma ** 2))
    fz = (1 / (2 * math.pi * sigma)) * np.exp((-zz ** 2) / (2 * sigma ** 2))

    return fx, fy, fz

# Generation of kernel onto grid
def generate_kernel(fx, fy, fz, sigma=1):
    xx,yy,zz = np.meshgrid(fx, fy, fz)
    
    return np.exp(- (xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

#Main Code- Testing
x, y, z = generate_grid()
gaussian_blur = gaussian_blur(x, y, z)
kernel = generate_kernel(*gaussian_blur)


