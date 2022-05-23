# -*- coding: utf-8 -*-
"""
ovlib.ebsd: EBSD data analysis
===============================================================================
"""

class ebsd(object):
    '''
    EBSD data object
    
    Parameters
    ----------
    orientations : 
    
        cryspy.xtal.orientation object
        
    x : numpy array of floats
    
        x locations of orientations
        
    y : numpy array of floats
    
        y locations of orientations
        
    phaseid : numpy array of ints
    
        phase identifiers
    
    Notes
    -----
    - At minimum, an EBSD object is an orientation object with a phase and
      x and y positions associated with each individual orientation.

    '''
    def __init__(self, orientations=[], x=[], y=[], phaseid=[]):
        # initialize the minimum amount of information in order to have EBSD
        # data. We need to have x and y coordinates and an orientation measured
        # at those coordinates.
        self.orientations = orientations
        self.x  = x
        self.y  = y
        self.phaseid = phaseid
    
    #def __repr__(self): 
    # TODO: Add __repr__ method
    #       Needs to show number of phases, number of data points, etc.

    def calc_scanstepsize(self):
        ''' adds step and scan size data to the ebsd object
        
        EBSD class method that uses x and y location information to add step
        size and scan size data to the ebsd object.
        '''
        import numpy as np
        nr = np.size(self.x)
        rows, cols = scansize(self.x, self.y)
        xstp, ystp = scanstep(self.x, self.y, rows, cols)
        idx  = scanrowlims(self.x, self.y, rows, cols)
        self.rows = rows
        self.cols = cols
        self.xstp = xstp
        self.ystp = ystp
        self.idx  = idx
        self.nr   = nr
    
    def prepare_for_plotting(self):
        ''' adds information to ebsd object required for plotting
        
        class method that adds information necessary for plotting to the object
        '''
        import numpy as np
        
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            print 'x and y coordinates are needed for plotting ebsd data.'
        
        if not hasattr(self, 'rows') or not hasattr(self, 'cols'):
            rows, cols = scansize(self.x, self.y)
            self.rows = rows
            self.cols = cols
            del rows, cols
        
        if not hasattr(self, 'xstp') or not hasattr(self, 'ystp'):
            xstp, ystp = scanstep(self.x, self.y, self.rows, self.cols)
            self.xstp = xstp
            self.ystp = ystp
            del xstp, ystp

        if not hasattr(self, 'idx'):
            idx  = scanrowlims(self.x, self.y, self.rows, self.cols)
            self.idx  = idx
            del idx
        
        if not hasattr(self, 'nr'):
            nr = np.size(self.x)
            self.nr = nr
            del nr

        # Kernel vertices and faces are needed for plotting similar to matlab's
        # patch function using matplotlib
        v, p = kernverts(self.x, self.y, self.rows, self.cols, 
                         self.xstp, self.ystp, self.idx)
        f    = kernfaces(v, p, self.idx, self.rows, self.cols, self.nr)
        self.v = v
        self.p = p
        self.f = f
        
        # For plotting using opengl, we need a different format of vertices and
        # faces
        tv, vf = triverts(self.nr, self.v, self.f, self.cols)
        self.tv = tv
        self.vf = vf        

#-------------------------------------------------------------------------------

def scansize(x, y):
    """ 
    scansize returns the number of rows and columns in oimdata on either
    a square or hexagonal grid. Two values for the # columns are
    returned in the case of a hex grid: the first for odd and the
    second for even. See hdrload for correct importing of the 
    data matrix.  
    
    Notes
    -----
    
    This one isn't as straightforward as you might assume.
    The data can come on either a hex grid or a square
    grid.  When on a hex grid, the data is staggered between
    rows, so the number of data points differs between rows,
    and the x-coordinate varies while the y-coordinate is
    constant within each row.
    
    If there is no header for the data, or if the header doesn't match the
    file contents, we can figure out this info from looking at the XY data.
    
    Ultimately, the old-school way I am using here to count
    the number of vertical pixels below isn't efficient but
    doesn't take very long.
    
    To get the oimdata input variable, see hdrload.
    
    If the input file is on a hex grid, columns returns first the odd
    then the even number.
    """ 
    import numpy as np

    if np.size(x) != np.size(y):
        print 'scansize: the lengths of x and y are not equal. this may result\
               in errors.'
    
    # count the number of pixels in the horizontal direction
    # i.e, the maximum number of columns 
    # (even numbered rows have fewer columns in the scan) and 
    # the first row (which is always ==0 in my TSL scans) is
    # by definition an odd row.
    cols1=np.sum(y==0) # cols in first row
    
    # if the first entry for the x-coordinate value in the first row
    # doesn't match the first entry for the x-coordinate value in the
    # second row, then it's hexagonal. This could maybe also be done
    # using the 'unique' function.
    if x[cols1] != x[0]:
       rows = np.sum(x==0) + np.sum(x==x[cols1])
       cols = np.array([1,2], dtype=np.uint32)
       cols[0]=cols1
       cols[1]=cols[0]-1
    else:
       nr  = np.size(x)
       rows= np.uint32(nr / cols1)
       cols= cols1

    return rows, cols
    
#-------------------------------------------------------------------------------

def scanstep(x, y, nr, nc):
    ''' finds the x and y step sizes of an ebsd scan
    
    Parameters
    ----------
    x : array
        x locations
    y : array
        y locations
    nr : scalar
        number of rows
    nc : array
        number of columns (can be 2 vector for hexagonal grid)
    
    Returns
    -------
    xstep : scalar
        step size in x direction
    ystep : scalar
        step size in y direction
    '''
    import numpy as np

    if np.size(x) != np.size(y):
        print 'scanstep: the lengths of x and y are not equal. this may result\
               in errors.'
    
    # Get the scan size
    if np.size(nc)==2: # hexagonal scan
        nc=nc[0];
    
    # Get the hex data x step size
    if nr<5:
        # we need at least two points in the row to get the step size
        # empirically (or we could read the header, but I think this method is
        # generally better in case the header doesn't match the data).
        # This means a hex scan size of at least 5 points is a reasonable
        # minimum: 2 points in the first and third rows, 1 point in the middle
        # row.
        print 'scanstep error: at least 5 data points required'
    else:
        xstep = x[1]-x[0]
        ystep = y[nc]-y[nc-1]
    
    return xstep, ystep

#-------------------------------------------------------------------------------

def scanrowlims(x, y, nrows, ncols):
    '''
    
    Notes
    -----
    SCANROWLIMS finds the indices that correspond to the left and right 
    limits of each row. Assumes a regular hexagonal or square grid.
    
    The algorithm below is slow, but since it typically only needs to be run
    once for a scan in a session, it seems ok for the time being.
    '''
    import numpy as np
    import cryspy.util as util

    if np.size(x) != np.size(y):
        print 'scanrowlims: the lengths of x and y are not equal. this may \
               result in errors.'
    
    if np.size(ncols)==2: # then we have a hex grid
    
        # Find the indices corresponding to the left and right edges of each row
        rowlims=np.zeros([nrows,2], dtype=np.uint32)
        
        l = 0
        for i in range(1, nrows+1):
            
            b = ~util.iseven(i)
            if b[0]:
                r = l + ncols[0] - 1
            else:
                r = l + ncols[0] - 2
                
            rowlims[i-1, 0] = l
            rowlims[i-1, 1] = r
            
            l = r + 1
            
    elif np.size(ncols)==1: # then we have a square grid
    
        rowlims=np.zeros([nrows,2])
        
        l = 0
        for i in range(1, nrows+1):
        
            r = l + ncols - 1
            
            rowlims[i-1, 0] = l
            rowlims[i-1, 1] = r
            
            l = r + 1
    
    return rowlims

#-------------------------------------------------------------------------------

def kernverts(x, y, rows, cols, xstp, ystp, idx, clipverts=True):
    '''
    KERNVERTS returns the grid of vertices corresponding to the boundaries of
    the data point kernels in an ebsd scan, as well as the indices corresponding
    to the edges of the rows in this grid. The results from the edges here
    differ from scanrowlims: In scanrowlims, we return the limits of the rows
    for the measurement grid. Here, we return the limits of the rows of the
    vertices grid.
        
    inputs: x and y coordinates of scan
    '''
    import numpy as np
    import cryspy.util as util
      
    if np.size(cols)==2:
        
        # Make hex grid
        xv1 = xstp / 2.0
        yv1 = ystp / np.sqrt(3.0)
        yv2 = yv1  / 2.0
        
        # first row, we need to cover the indices that are above the face nodes
        v = np.zeros([(rows+1)*(2*cols[0]+1), 2], dtype=float)
        p = np.zeros([2 * rows + 2, 2], dtype=np.uint32)
        yy = y[[idx[0][0]]]
        v1 = cols[0]
        v2 = cols[0] + 1
        xvs1 = np.arange(1, v1+1) * xstp - xstp
        xvs2 = np.arange(1, v2+1) * xstp - xstp
        v[0:v1, 0] = xvs1
        v[0:v1, 1] = np.tile(yy - yv1, v1)
        p[0, 0] = 0
        p[0, 1] = v1-1
        
        v[v1:v1+v2, 0] = xvs2 - xv1
        v[v1:v1+v2, 1] = np.tile(yy - yv2, v2)
        p[1, 0] = v1
        p[1, 1] = v1+v2-1
        
        # All other nodes
        nV=v1+v2
        j=2
        for i in range(1, rows+1):
            yy = y[idx[i-1][0]]
            v1 = cols[0]
            v2 = cols[0] + 1
            xvs1 = np.arange(1, v1+1) * xstp - xstp
            xvs2 = np.arange(1, v2+1) * xstp - xstp
            
            if ~util.iseven(i):
    
                v[nV:nV+v2, 0] = xvs2-xv1
                v[nV:nV+v2, 1] = np.tile(yy+yv2, v2)
                p[j, 0] = nV
                p[j, 1] = nV+v2-1
                j += 1
                v[nV+v2:nV+v2+v1, 0] = xvs1
                v[nV+v2:nV+v2+v1, 1] = np.tile(yy+yv1, v1)
                p[j, 0] = nV+v2
                p[j, 1] = nV+v2+v1-1
                j += 1
                nV=nV+v1+v2
                
            else:
                
                v[nV:nV+v1, 0] = xvs1
                v[nV:nV+v1, 1] = np.tile(yy+yv2, v1)
                p[j, 0] = nV
                p[j, 1] = nV+v1-1
                j += 1
                v[nV+v1:nV+v1+v2, 0] = xvs2-xv1
                v[nV+v1:nV+v1+v2, 1] = np.tile(yy+yv1, v2)
                p[j, 0] = nV+v1
                p[j, 1] = nV+v1+v2-1
                j += 1
                nV=nV+v1+v2
        
    elif np.size(cols)==1: # square grid
    
        v = np.zeros([(cols+1)*(rows+1), 2], dtype=float)
        p = np.zeros([rows+1, 2], dtype=np.uint32)
        
        v[:,0] = np.tile(xstp*np.arange(0,cols+1),rows+1)-xstp/2.0
        v[:,1] = np.reshape(np.tile(ystp*np.arange(0,rows+1),
                                    (cols+1,1)).T-ystp/2.0, -1)
        
        p[:,0] = np.arange(1,(rows+1)*(cols+1),cols+1)-1
        p[:,1] = np.arange(1,(rows+1)*(cols+1),cols+1)-1+cols
    
    if clipverts==True and np.size(cols)==2:
        # Fix the top row of points
        v[p[0,0]:p[0,1]+1][:,1] = np.tile(y[idx[0,0]]-np.sqrt(np.spacing(1)),
                                                             (p[0,1]-p[0,0]+1))
        v[p[1,0]:p[1,1]+1][:,1] = np.tile(y[idx[0,0]], (p[1,1]-p[1,0]+1))
        
        # Fix the bottom row of points
        ploc = np.shape(p)[0]-1
        iloc = np.shape(idx)[0]-1
        v[p[ploc,0]:p[ploc,1]+1][:,1] = np.tile(y[idx[iloc,0]] + 
                                             np.sqrt(np.spacing(1)),
                                                       (p[ploc,1]-p[ploc,0]+1))
        v[p[ploc-1,0]:p[ploc-1,1]+1][:,1] = np.tile(y[idx[iloc,0]], 
                                                   (p[ploc-1,1]-p[ploc-1,0]+1))
        
        # Fix the left and right columns of points
        for i in range(0,np.size(p[:,0])-1):
            valmin = np.amin(x)-np.sqrt(np.spacing(1))
            valmax = np.amax(x)+np.sqrt(np.spacing(1))
            v[p[i,0]][0] = valmin
            v[p[i,1]][0] = valmax
        
    
    return v, p # v is the array of vertices, p is the array of edge indices
        
#------------------------------------------------------------------------------

def kernfaces(v, p, idx, rows, cols, nr):
    '''
    KERNFACES associates the faces of the kernels with their surrounding
    vertices, for both hexagonal and square grid patterns.
    ''' 
    import numpy as np
    import cryspy.util as util
    
    # Associate faces with vertices
    ndx=0;
    
    if np.size(cols)==2: # hex grid

        pb = util.progbar(finalcount=nr, 
                     message='ASSOCIATING FACES WITH VERTICES')   
    
        f=np.zeros([nr,6], dtype=np.uint32)    
        for i in range(1,rows+1):
            
            j=(i-1)*2+1
    
            if ~util.iseven(i): # odd row
                
                tmp=np.array([
                     np.arange( p[j-1,0]  , p[j-1,1]+1),
                     np.arange( p[j  ,0],   p[j  ,1]  ),
                     np.arange( p[j  ,0]+1, p[j  ,1]+1),
                     np.arange( p[j+1,0],   p[j+1,1]  ),
                     np.arange( p[j+1,0]+1, p[j+1,1]+1),
                     np.arange( p[j+2,0],   p[j+2,1]+1)   ]).T
                     
            else: # even row
    
                tmp=np.array([
                     np.arange(p[j-1,0]+1,  p[j-1,1]  ),
                     np.arange(p[j  ,0],    p[j  ,1]  ),
                     np.arange(p[j  ,0]+1,  p[j  ,1]+1),   
                     np.arange(p[j+1,0],    p[j+1,1]  ), 
                     np.arange(p[j+1,0]+1,  p[j+1,1]+1),   
                     np.arange(p[j+2,0]+1,  p[j+2,1]  )     ]).T
    
    
            ntmp = np.size(tmp)/6
            
            for k in range(0, ntmp):
                f[ndx, 0:6] = [tmp[k,0], tmp[k,1], tmp[k,3], tmp[k,5], tmp[k,4], tmp[k,2]]
                ndx += 1
    
            pb.update(i)
            
        pb.update(-1)
        
    elif np.size(cols)==1: # square grid
    
        f  = np.zeros([nr, 4], dtype=np.uint32)
        f1 = np.arange(0, cols, dtype=np.uint32)
        f2 = f1 + 1
        for i in range(0,rows):
            f3 = np.arange(0, cols, dtype=np.uint32) + (cols + 1) * (i + 1)
            f4 = f3 + 1
            f[idx[i,0]:idx[i,1]+1, 0] = f3
            f[idx[i,0]:idx[i,1]+1, 1] = f1
            f[idx[i,0]:idx[i,1]+1, 2] = f2
            f[idx[i,0]:idx[i,1]+1, 3] = f4    
            f1 = f3
            f2 = f4
    
    return f

#------------------------------------------------------------------------------

def triverts(n, v, f, cols):
    #TODO: Documentation!
    
    import numpy as np
        
    # Add third dimension to vertex data
    tv = np.zeros([np.size(v)/2, 3])
    tv[:,0] = v[:,0]
    tv[:,1] = v[:,1]
    del v
    
    # Create the face array
    if np.size(cols)==2: # hex grid
        
        vf = np.zeros([n*4,3], dtype=np.uint32)
        k  = 0
        for i in range(0,n):
            
            # triangle 1
            vf[k,0] = f[i,0]
            vf[k,1] = f[i,1]
            vf[k,2] = f[i,5]
            k += 1
            
            # triangle 2
            vf[k,0] = f[i,5]
            vf[k,1] = f[i,1]
            vf[k,2] = f[i,4]
            k += 1
        
            # triangle 3
            vf[k,0] = f[i,2]
            vf[k,1] = f[i,4]
            vf[k,2] = f[i,1]
            k += 1
        
            # triangle 4
            vf[k,0] = f[i,3]
            vf[k,1] = f[i,4]
            vf[k,2] = f[i,2]
            k += 1
    
    else: # square grid

        vf = np.zeros([n*2,3], dtype=np.uint32)
        k  = 0
        for i in range(0,n):
            
            # triangle 1
            vf[k,0] = f[i,1]
            vf[k,1] = f[i,2]
            vf[k,2] = f[i,0]
            k += 1
            
            # triangle 2
            vf[k,0] = f[i,2]
            vf[k,1] = f[i,3]
            vf[k,2] = f[i,0]
            k += 1

    return tv, vf

#------------------------------------------------------------------------------


        
    
    
    

    
