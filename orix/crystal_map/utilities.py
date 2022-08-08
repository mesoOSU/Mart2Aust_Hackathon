#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:08:49 2022

@author: paytone
"""
import numpy as np
import time
import networkx as nx
import getopt
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from numpy import arctan2, cumsum, matrix, squeeze, unique, linspace,zeros
from statistics import mean
from math import sqrt, floor, copysign
from orix.quaternion import Orientation, Rotation, symmetry, Misorientation
from orix.quaternion.rotation import Rotation
import orix.vector as vect
from orix.utilities import utilities
from orix.utilities.utilities import sortrows, regularPoly, uniquerows
from matplotlib import path
import matplotlib.pyplot as plt
import math
import warnings
import alphashape

def spatial_decomposition(X, unit_cell=None, boundary='hull', qhull_opts='Q5 Q6 Qs'):
    '''
    % decomposite the spatial domain into cells D with vertices V,
    %
    % Output
    %  V - list of vertices
    %  F - list of faces
    %  I_FD - incidence matrix between faces to cells

    % compute voronoi decomposition
    % V - list of vertices of the Voronoi cells
    % D   - cell array of Vornoi cells with centers X_D ordered accordingly
    '''

    if unit_cell.all() == None:
        unit_cell = calcUnitCell(X)
        
    # compute the vertices
    [V, faces] = generateUnitCells(X, unit_cell)
    # NOTE: V and faces do not give exact values as compared to the MATLAB
    # implementation. However, Eric and Rohan have confirmed that V and faces
    # at least have the same shape as the MATLAB implementation.

    D = np.empty(len(X), dtype=object)

    for k in range(X.shape[0]):
        D[k] = faces[k, :]

    else:
        dummy_coordinates = calcBoundary(X, unit_cell, boundary)

        vor = Voronoi(np.vstack([X, dummy_coordinates]), qhull_options=qhull_opts)  # ,'QbB'

        V = vor.vertices                            # vertices of the voronoi diagram
        D = vor.regions                             # list of faces of the Voronoi cells
        D = D[0:X.shape[0]]                         # Cut off the dummy coordinates

    # now we need some adjacencies and incidences
    iv = np.hstack(D)                       # nodes incident to cells D
    iid = zeros(len(iv), dtype=np.int64)    # number the cells

    # Some MATLAB stuff goin on here... : p = [0; cumsum(cellfun('prodofsize',D))];
    D_prod = matlab_prod_of_size(D)
    p = np.hstack([0, np.cumsum(D_prod)-1])

    # original matlab: for k=1:numel(D), id(p(k)+1:p(k+1)) = k; end
    for k in range(len(D)):
        iid[p[k]+1 : p[k+1]] = k

    # next vertex
    # indx = 2:numel(iv)+1;
    ind_x = np.arange(1,len(iv)+1)
    # indx(p(2:end)) = p(1:end-1)+1;
    ind_x[p[1:]] = p[0:-1] + 1
    # ivn = iv(indx);
    iv_n = iv[ind_x]

    # edges list
    F = np.vstack([iv[:], iv_n[:]]).T

    # should be unique (i.e one edge is incident to two cells D)
    F, ia, ie = uniquerows(F)   # numpy indexing arrays such that edges = F[ia] and F = edges[ie]

    # faces incident to cells, F x D
    #original matlab: I_FD = sparse(ie,id,1); 
    # Matlab stores as csr matrix, so we use this class below
    data = np.ones(np.shape(ie))
    I_FD = csr_matrix((data, (ie, iid)))

    #I_FD = csr_matrix( (data, (ie, iid)), shape=[ie.shape[0], ie.shape[0]]) # could also use csc_matrix() if it improves
                                # performance !
    '''
    NOTE: may need to use different sparse matrix. The matrix is incorrect as
          compared to the MATLAB implementation.
          
          Could also be causing problems in the Householder matrix initialization.
    '''

    return V, F, I_FD


def calcBoundary(X, unit_cell, boundary='hull'):
    '''
    dummy coordinates so that the voronoi-cells of X are finite

    Inputs:
    --------------
    X : n x 2 numpy array of [x,y] vertex coordinates

    unit_cell : n x 2 numpy array of [x,y] coordinates of "pixel" boundary (e.g. hexagon or square)

    var_arg_in : ???
    ???


    Outputs:
    --------------
    ??? : ???
    ???

    Dependencies:
    --------------
    import getopt
    import numpy as np
    from scipy.spatial import ConvexHull
    from numpy import arctan2, diff, cumsum, matrix, squeeze, unique
    from statistics import mean
    from math import sqrt, floor, copysign
    '''
    
    dummy_coordinates = []

    if boundary.isalpha():
        
        if boundary.lower() == 'tight':
            
            k = alphashape.alphashape(X, 0.95) #TODO: 0.95 matches mtex, but this would be a good user variable
            k2 = list(zip(*k.exterior.coords.xy))
            locations = []
            for item in k2:
                locations.append(np.where(np.all(item==X, axis=1)))
            locations = np.squeeze(locations)
            
            bounding_X  = erase_linearly_dependent_points(X, locations)
            
            #Testing
            #import matplotlib.pyplot as plt
            #plt.figure()
            #lt.plot(bounding_X[:,0], bounding_X[:,1])
            #plt.show()

        elif (boundary.lower() == 'hull' or
            boundary.lower() == 'convexhull'):
            
            k = ConvexHull(X)
            k2 = k.vertices
            
            bounding_X  = erase_linearly_dependent_points(X, k2)
            
            #Testing
            #import matplotlib.pyplot as plt
            #plt.figure()
            #plt.plot(bounding_X[:,0], bounding_X[:,1])
            #plt.show()
            
        elif boundary.lower() == 'cube':
            # set up a rectangular box
            envelope_X = [np.amin(X), np.amax(X)]
            bounding_X = [
                envelope_X[1], envelope_X[3],
                envelope_X[2], envelope_X[3],
                envelope_X[2], envelope_X[4],
                envelope_X[1], envelope_X[4],
                envelope_X[1], envelope_X[3],
                ]

        else:
            raise ValueError('Unknown boundary type. Available options are \
            ''tight'',''hull'', ''convexhull'' and ''cube''.')

    elif isinstance(boundary, float):
        bounding_X = boundary
    
    radius = np.mean(np.sqrt(np.sum( (unit_cell**2), axis=1)))
    
    edge_length = ((matlabdiff(bounding_X)**2).sum(axis=1, keepdims=True))**0.5 
    #edgeLength = sqrt(sum(diff(boundingX).^2,2));
    
    # fill each line segment with nodes every 20 points (in average)
    nto = np.int32(np.fix( (edge_length>0)*4 ))

    #nto = floor( edge_length*(2*radius) )
    #nto = fix((edgeLength>0)*4); fix(edgeLength*(2*radius));

    csum = np.int32(cumsum(np.vstack([0, nto+1])))

    bounding_X = left_hand_assignment(bounding_X, csum)
    
    # interpolation
    for k in range(nto.shape[0]):
        for dim in range(nto.shape[1]+1):
            bounding_X[csum[k]:csum[k+1]+1, dim] = linspace(
                bounding_X[csum[k],dim],
                bounding_X[csum[k+1], dim],
                nto[k,0]+2
            )

    # homogeneous coordinates
    X = np.hstack([X, np.ones([X.shape[0],1])])
    bounding_X = np.hstack([bounding_X, np.ones([bounding_X.shape[0],1])])

    # direction of the edge
    edge_direction = matlabdiff(bounding_X)
    edge_angle = arctan2( edge_direction[:, 1], edge_direction[:,0] )
    edge_length = np.sqrt( (edge_direction**2).sum(axis = 1, keepdims=True))

    # shift the starting vertex
    r = Rotation.from_axes_angles(vect.Vector3d([0, 0, 1]), edge_angle)
    a = vect.Vector3d([0, radius, 1])
    b_X = np.squeeze(r * a)
    offset_X = np.squeeze(b_X.data) - np.squeeze(bounding_X[0:-1, :])
 
    dummy_coordinates = np.array([[0,0]])
    for k in range(bounding_X.shape[0]-1):

        # mirror the point set X on each edge
        p_X = np.matmul(X,
            -1*( 
                np.matmul(
                    np.matmul(
                        translationMatrix(offset_X[k, :]),
                        householderMatrix(edge_direction[k, :])
                    ), 
                    translationMatrix(offset_X[k, :])
                ).T
            )
        )

        # distance between original and mirrored point
        dist = np.sqrt(np.sum((X[:,0:-1]-p_X[:,0:-1])**2, axis=1))

        intend_X = 2. * radius * np.sign(edge_direction[k,0:2])

        # now try to delete unnecessary points
        m = 2        
        
        # NEED TO REBUILD LOOP STRUCTURE HERE FOR PARITY - JC
        #tmp_X = p_X[np.argwhere(dist < m*radius), 0:-1]
        
        while(True):

            tmp_X = p_X[dist < m*radius, 0:2]

            right = np.matmul(tmp_X - np.tile( bounding_X[k, 0:2]   - intend_X, [np.shape(tmp_X)[0], 1]), edge_direction[k, 0:2].T) < 0
            left  = np.matmul(tmp_X - np.tile( bounding_X[k+1, 0:2] + intend_X, [np.shape(tmp_X)[0], 1]), edge_direction[k, 0:2].T) > 0

            tmp_X = tmp_X[~np.any(np.vstack([right, left]), axis=0), 0:2]
            
            with np.errstate(divide='ignore'):
                if edge_length[k] / tmp_X.shape[0] < radius/3:
                    break
                elif m < 2**7:
                    m = m*2
                elif m < 2**7+100:
                    m = m+10
                else:
                    break
        
        if tmp_X.size > 0:
            dummy_coordinates = np.vstack([dummy_coordinates, tmp_X])
            

    # Since we initialized dummy_coords with a 0 row, pop it out
    dummy_coordinates = np.delete(dummy_coordinates, 0, axis=0)
    
    # MATLAB implementation used the standard "unique" on a float array
    # Enforced a unique rows with a tolerance of 1e-12 which agrees with
    # MATLAB's "uniquetol" output
    dummy_coordinates = np.unique(dummy_coordinates.round(decimals=12), axis=0)
    
    # Use matplotlib path to check if we left any points in/on the polygon
    # Note, python uses 'FALSE' to delete rows, so inverse the logic returned.
    p = path.Path(bounding_X[:, 0:2])
    dummy_coordinates = dummy_coordinates[~p.contains_points(dummy_coordinates)]
    
    return dummy_coordinates

# ----------------------------------------------------------------------------

def matlab_prod_of_size(D):
    '''
    Testing:
    test1 = [[1, 2, 3], [1, 2], [1]]
    matlab_prod_of_size(test1)
    '''
    prod_of_size = []
    for elem in D:
        prod_of_size.append(len(elem))
    return prod_of_size

# ----------------------------------------------------------------------------

def erase_linearly_dependent_points(X, k):
    '''
    subfunction to remove linearly dependent points.

    Inputs:
    --------------
    X : n x 2 numpy array of coordinates
    k : list of coordinate pairs corresponding to the convex or concave hull

    Outputs:
    --------------
    boundingX : n x 2 numpy array for bounding box

    Dependencies:
    --------------
    from scipy.spatial import ConvexHull
    '''

    # erase all linear dependent points
    angle = np.arctan2(X[k[0:-1],0]-X[k[1:],0],
        X[k[0:-1],1]-X[k[1:],1])
    test = np.abs(np.diff(angle))>np.spacing(1.0)
    k2 = k[np.concatenate([[True], test, [True]])]
    
    # Build the coord set of the convex hull
    # Pad on the end->start connection to close hull
    # If the origin is contained as a boundary point
    # then we need to shift the array values so the origin
    # is the end->start wrap
    boundingX = X[k2,:]
    
    zero_row = -1
    for i in range(boundingX.shape[0]):
        if np.all(boundingX[i]==0):
            zero_row = i
    
    if (zero_row != -1):
        boundingX = np.roll(boundingX, -zero_row, axis=0)
    
    boundingX = np.vstack([boundingX , boundingX[0,:] ])
 
    return boundingX

# ----------------------------------------------------------------------------

def left_hand_assignment(X, a):
    '''
    Attempts to replicate MATLAB left-hand assignment for a 2D array.
 

    Inputs:
    --------------
    X : 2D numpy array
 

    Outputs:
    --------------
    X : 2D numpy array
        This is different from the X passed in, it will return a larger array with
        more zeros.
 
    Dependencies:
    --------------
    import numpy as np
    '''
    
    if a.dtype != 'int32' or 'int64':
        warnings.warn('parameter ''a'' must be of integer type. Converting ''a'' into integers and moving on...')

    a = np.int32(a)

    bound_X_init = X
    ncols = X.shape[1]
 
    max_a = np.amax(a)
    bound_X_fin = np.zeros([max_a+1, ncols])
  
    bound_X_fin[0:bound_X_init.shape[0], :] = bound_X_init

    for i, elem in enumerate(a[0:-1]):
        bound_X_fin[elem, :] = bound_X_init[i, :]

    return bound_X_fin

# ----------------------------------------------------------------------------

def matlabdiff(myArray):
    return myArray[1:,:] - myArray[0:-1,:]

# ----------------------------------------------------------------------------

def gbc_angle(q, CS, D_l, D_r, threshold=5.):
    '''
    THIS IS AN INITIAL DRAFT TRANSLATION OF GMTEX's BC_ANGLE
    
    
    # the inputs are: quaternion(ebsd.rotations),ebsd.CSList{p},Dl(ndx),Dr(ndx),gbcValue(p),varargin{:})

    
    #% FOR TESTING
    ################## load materials and image 
    # this whole section will be replaced by graph cut function eventually
    path = r'/Users/paytone/Documents/GitHub/maxflow_for_matsci/Data/steel_ebsd.ang'
    
    # Read each column from the file
    euler1, euler2, euler3, x, y, iq, dp, phase_id, sem, fit  = np.loadtxt(path, unpack=True)
    
    phase_id = np.int32(phase_id)
    
    # Create a Rotation object from Euler angles
    euler_angles = np.column_stack((euler1, euler2, euler3))
    rotations = Rotation.from_euler(euler_angles)
    
    # Create a property dictionary
    properties = dict(iq=iq, dp=dp)
    
    # Create unit cells of the phases
    structures = [
        Structure(
            title="ferrite",
            atoms=[Atom("fe", [0] * 3)],
            lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
        ),
    ]
    phase_list = PhaseList(
        names=["ferrite"],
        point_groups=["432"],
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
    
    import numpy
    
    #% get pairs of neighbouring cells {D_l,D_r} in A_D
    #A_D = I_FD'*I_FD==1;
    #[Dl,Dr] = find(triu(A_D,1));
    '''
    
    # convert threshold into radians
    threshold = threshold * np.pi / 180.
    
    
    # now check whether the have a misorientation heigher or lower than a threshold
    o1 = q[D_l] # orientation of node u
    o2 = q[D_r] # orientation of node v
    m = Misorientation(o1 * o2.conj) # misorientations between every u and v
    m.symmetry = (CS, CS) # Oh is symmetry (need to un-hard code)
    m = m.map_into_symmetry_reduced_zone() #TODO: The result of this doesn't actually make sense. Why are there two rows?
    
    criterion = np.abs(m[0,:].angle) > np.cos(threshold/2.0)
    
    # Part of mtex code; purpose not clear:
    #if np.any(~criterion):
    #  qcs = symm_l.proper_subgroup #TODO: Look into whether this actually makes sense, would be best to use the same symmetry for both L and R but it wasn't clear how to do this in orix at this time
    #  criterion[~criterion] = np.amax(np.abs(Rotation.dot_outer(m[~criterion],qcs)), axis=1) > np.cos(threshold/2.);
    
    # Here is how mtex gets criterion using orientations, not yet possible in orix
    # o_Dl = orientation(q(Dl),CS,symmetry);
    # o_Dr = orientation(q(Dr),CS,symmetry);
    # criterion = dot(o_Dl,o_Dr) > cos(threshold/2);
    
    return criterion

# ----------------------------------------------------------------------------

def householderMatrix(v):
    '''
    Parameters
    ----------
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.
        
    For Testing:
    H_out = np.loadtxt('householder_output.txt', delimiter=',')
    H_in = np.loadtxt('householder_input.txt', delimiter=',')
    H_test = householderMatrix(H_in)

    '''
    # H = @(v) eye(3) - 2./(v(:)'*v(:))*(v(:)*v(:)') ;
    v = np.atleast_2d(v)
    H = np.eye(3) - 2. / np.matmul(v, v.T) * np.matmul(v.T, v)
    return H

# ----------------------------------------------------------------------------

def translationMatrix(s):
    '''
    Parameters
    ----------
    v\s : TYPE
        DESCRIPTION.

    Returns
    -------
    T : TYPE
        DESCRIPTION.
        
    For Testing:
    T_out = np.loadtxt('translation_output.txt', delimiter=',')
    T_in = np.loadtxt('translation_input.txt', delimiter=',')
    T_test = translationMatrix(T_in)
    '''
    T  = np.array([[ 1., 0., s[0]],[0., 1., s[1]],[0., 0., 1.]])
    return T

# ----------------------------------------------------------------------------

def generateUnitCells(xy, unitCell):
    '''
    % generates a list of patches according to spatial coordinates and the unitCell
    %
    % Inputs
    %  xy       - midpoints of the cells
    %  unitCell - spatial coordinates of the unit cell
    %
    % Parameters
    %  v     - list of vertices
    %  faces - list of faces
    
    # For Testing:
    xy = np.loadtxt('spatialDecomposition_input_X.csv', delimiter=',')
    unitCell = np.loadtxt('spatialDecomposition_input_unitCell.csv', delimiter=',') 
    v, faces = generateUnitCells(xy, unitCell)  
    randomface = np.random.randint(np.size(xy[:,0]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(v[faces[randomface,:],0], v[faces[randomface,:],1], '--')
    plt.axis('equal')
    plt.show
    '''
    def clockwise_sort(points_list):

        def angle_to(point):
        # assumes already mapped to (0,0) center
        # modify as needed to get result in desired range
            return math.atan2(point[1],point[0])

        sorted_points = sorted(points_list, key=angle_to, reverse=True)
        sorted_points = np.roll(sorted_points, 1, axis=0)

        return sorted_points

    unitCell = clockwise_sort(unitCell)

    # add unitCell values to data.
    def add_unitcell_values(X):
        X_new = []
        for elem1 in unitCell:
            for elem2 in X:
                X_new.append(list(elem1+elem2))

        X_new = np.array(X_new)
        return X_new
        
    X_new = add_unitcell_values(xy)

    x = X_new[:, 0]
    y = X_new[:, 1]

    # remove equal points
    eps = np.amin([np.sqrt(np.diff(unitCell[:,0])**2 + np.diff(unitCell[:,1])**2)])/10.;

    verts = np.squeeze(np.round([x - np.amin(x), y - np.amin(y)]/eps)).T

    v, m, n = utilities.uniquerows(verts)
    v, order = sortrows(v, return_order=True)

    x = np.squeeze(x)
    y = np.squeeze(y)
    m = np.squeeze(m)


    v = np.vstack([x[m][order], y[m][order]]).T

    # set faces
    faces = np.reshape(order[n], [-1,np.size(unitCell[:,1])])
    
    return v, faces

def calcUnitCell(xy, *args):
    '''
    Compute the unit cell for an EBSD data set
    '''
    # isempty return
    ## NOT WRITTEN
    
    # Check if the input xy is 3-col data and do some recursion
    ## NOT WRITTEN
    
    # Make the first estimate of the grid resolution
    area = (np.max(xy[:,0])-np.min(xy[:,0]))*(np.max(xy[:,1])-np.min(xy[:,1]))
    dxy = np.sqrt(area / np.max(xy.shape))
    
    # Compensate for the single line EBSD
    ## NOT WRITTEN
    
    # Remove the duplicates from the coords
    tol = 0.010 / np.sqrt(xy.shape[0])
    xy = np.unique(xy.round(decimals=6), axis=0) # Need better solution here for uniquetol
    # see https://stackoverflow.com/questions/46101271/what-does-uniquetol-do-exactly
    
    # Reduce the data set
    if (np.max(xy.shape) > 10000):
        xy = subSample(xy, 10000)
    
    # Wrap the Voronoi decomp in a try/catch
    try:
        vor = Voronoi(xy, qhull_options='Qz')
        V = vor.vertices
        D = vor.regions

        # Some notes here: Our V omits the [inf inf] first cell that MATLAB returns
        # but our D does contain the first dead cell that MATLAB omits

        # Try the sorting thing from:
        # https://stackoverflow.com/questions/59090443/how-to-get-the-same-output-of-voronoin-of-matlab-by-scipy-spatial-voronoi-of-pyt

        # Find point coordinate for each region
        sorting = [np.where(vor.point_region==x)[0][0] for x in range(1, len(D))]
        # sort regions along coordinate list `sorting` (exclude first, empty list [])
        sorted_regions = [x for _, x in sorted(zip(sorting, D[1:]))]
        # The above command gives us parity with MATLAB's voronoin
        # but the IDs of the vertices aren't necessarily the same

        # Compute the area of each Voronoi cell
        areaf = np.zeros((np.max(xy.shape)))
        for i, indices in enumerate(sorted_regions):

            # Skip this calculation if open cell
            if -1 in np.array(indices):
                areaf[i] = np.nan          
                continue

            # Retrieve the vertices in this Voronoi cell
            # and compute the area with a shoelace formula
            r_verts = np.vstack([V[indices,:], V[indices[0],:]])
            areaf[i] = polyArea(r_verts[:,0],r_verts[:,1])

        # Now, find the "unit cell" which is the Voronoi cell with the 
        # smallest area. If there are duplicates, grab the first.
        ci = int(np.argwhere(areaf==np.nanmin(areaf))[0])

        # Get the vertices of the "unit cell"
        unitCell = np.array([V[sorted_regions[ci],0] - xy[ci,0], \
                             V[sorted_regions[ci],1] - xy[ci,1]]).transpose()

        # Ignore some points given the grid resolution value
        # It would be helpful is "ignore" was returned as a tuple instead of 2d array
        ignore = np.hstack([np.zeros((1, 1), dtype=bool), [np.sum(np.power(matlabdiff(unitCell),2), axis=1) < dxy/100]])
        unitCell = unitCell[~ignore[0,:],:] # Inversion here since Python deletes False

        # Check if this is a regular polygon
        if (isRegularPoly(unitCell)):      # Missing varargin here
            return unitCell

    except:
        pass

    ## EVERYTHING BELOW HERE IS IF THE VORONOI UNITCELL DOESNT 
    ## END UP CREATING A REGULAR POLYGON UNITCELL
    ## Please code me :)
    
    # Get the second estimate of the grid resolution ?
    ## NOT WRITTEN YET
    
    # Get some grid options
    ## NOT WRITTEN YET - REQUIRES VARARGIN
    
    # Otherwise, do some handling to return a regular unit cell
    # if the previous Voronoi decomp did not
    ## NOT WRITTEN YET - REQUIRES regularPoly to be ported
    
    # return unitCell
    
def subSample(xy, N):
    '''
    Find a square subset of about N points
    '''
    
    xminmax = np.vstack([np.min(xy[:,0]), np.max(xy[:,0])])
    yminmax = np.vstack([np.min(xy[:,1]), np.max(xy[:,1])])

    while (np.max(xy.shape) > N):
        
        if (matlabdiff(xminmax) > matlabdiff(yminmax)):
            xminmax = np.divide(np.matmul(np.array([[3, 1], [1, 3]]), xminmax), 4)
        else:
            yminmax = np.divide(np.matmul(np.array([[3, 1], [1, 3]]), yminmax), 4)
        
        log_a = np.logical_and(xy[:,0] > xminmax[0], (xy[:,0] < xminmax[1]))
        log_b = np.logical_and(xy[:,1] > yminmax[0], (xy[:,1] < yminmax[1]))
        log_ab = np.logical_and(log_a, log_b)
        
        xy = xy[log_ab, :]

    return xy    
        
def polyArea(x,y):
    return 0.50*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def isRegularPoly(unitCell):
    # Compute the side lengths of all the "edges"
    sideLength = np.sqrt(np.sum(np.power(unitCell,2), axis=1))
    sides = sideLength.size
    
    # Some complex conversions and a roll
    uC = unitCell[:,0] + unitCell[:,1] * 1.0j
    nC = np.roll(uC, -1, axis=0)
    
    # Determine the angles between the edges
    enclosingAngle = np.divide(uC, nC)
    enclosingAngle = np.abs(np.real(enclosingAngle)) + \
                     np.abs(np.imag(enclosingAngle)) * 1.0j

    # If it is extremely close to a square or hex, we are good to return
    isRegular = (np.any(sides==np.array([4, 6]))) and \
                (np.linalg.norm(enclosingAngle - np.mean(enclosingAngle)) < np.deg2rad(0.05))
    
    return isRegular