# -*- coding: utf-8 -*-
"""
ovlib.util: Utilities
===============================================================================
"""

def vecarraycross(b, c):
    ''' cross products of arrays of vectors '''
    import numpy as np
    bx = vecarrayconvert(b[0])
    by = vecarrayconvert(b[1])
    bz = vecarrayconvert(b[2])
    cx = vecarrayconvert(c[0])
    cy = vecarrayconvert(c[1])
    cz = vecarrayconvert(c[2])
    
    if np.shape(bx) == np.shape(by) == np.shape(bz) == \
       np.shape(cx) == np.shape(cy) == np.shape(cz):
        
        ax = by * cz - bz * cy
        ay = bz * cx - bx * cz
        az = bx * cy - by * cx
        
        return [ax, ay, az]
    else:
        print "vecarraycross error: check that the lengths of arguments are equal."
        
#-------------------------------------------------------------------------------
    
def vecarraynorm(a):
    ''' norm of an array of vectors '''   
    import numpy as np
    ax = vecarrayconvert(a[0])
    ay = vecarrayconvert(a[1])
    az = vecarrayconvert(a[2])
    
    if np.shape(ax) == np.shape(ay) == np.shape(az):     
        nrm=(ax*ax + ay*ay + az*az)**0.5
    else:
        print "vecarraynorm error: check that the lengths of arguments are equal."
        
    return nrm

#-------------------------------------------------------------------------------

def vecarraydot(b,c):
    ''' dot products of an array of vectors  '''
    import numpy as np
    bx = vecarrayconvert(b[0])
    by = vecarrayconvert(b[1])
    bz = vecarrayconvert(b[2])
    cx = vecarrayconvert(c[0])
    cy = vecarrayconvert(c[1])
    cz = vecarrayconvert(c[2])   

    if np.shape(bx) == np.shape(by) == np.shape(bz) == \
       np.shape(cx) == np.shape(cy) == np.shape(cz):
           
        return bx*cx + by*cy + bz*cz

    else:

        print "vecarraydot error: check that the lengths of arguments are equal."

#-------------------------------------------------------------------------------
  
def vecarraygcd(a,b):
    ''' Compute greatest common denominator for values in two arrays'''    
    import numpy as np
    
    a = vecarrayconvert(a)
    b = vecarrayconvert(b)

    if np.shape(a) == np.shape(b):
                              
        r = np.zeros( np.size(a) ) # preallocate remainder array

        # if b is smaller than floating point error, we will make gcd=1 
        a[ b < np.spacing(1) ] = 1.0
        j = b > np.spacing(1)

        if j.any():
            while j.any():
                r[j] = a[j] % b[j]
                a[j] = b[j]
                b[j] = r[j]
                j[b==0]=False
                
            return a
        else:
            return a

#-------------------------------------------------------------------------------

def vecarrayconvert(a): # FIXME: Not necessary? Can just use asanyarray?
    ''' convert any reasonable datatype into an n x 3 vector array'''
    import numpy as np
    
    #a = np.asanyarray(a)
    
    #a = np.squeeze( np.float64( list( (a,) )))    
    #a = np.atleast_2d(np.asarray(a))

    # Make the arrays indexable if they are actually scalars
    #if np.size(a) == 1:
    #    a = np.array([a])

    a = np.atleast_1d(np.squeeze(np.asanyarray(a)))
    
    return a

#-------------------------------------------------------------------------------

def stereotrans(xyz, center=[0,0,1], south=[0,1,0], east=[1,0,0]):
    '''
    Perform non-standard stereographic projections following 
    Kosel TH, J Mater Sci 19 (1984)
    '''
    import numpy as np
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed) 
    
    xyz = vecarrayconvert(xyz)
    nrmxyz = 1.0/vecarraynorm(xyz)
    
    xx = xyz[0]*nrmxyz
    yy = xyz[1]*nrmxyz
    zz = xyz[2]*nrmxyz

    n = np.size(xx)     
    
    aa = np.tile(abc[0]*nrmabc,n)
    bb = np.tile(abc[1]*nrmabc,n)
    cc = np.tile(abc[2]*nrmabc,n)
    dd = np.tile(fed[0]*nrmdef,n)
    ee = np.tile(fed[1]*nrmdef,n)
    ff = np.tile(fed[2]*nrmdef,n)
    uu = np.tile(uvw[0]*nrmuvw,n)
    vv = np.tile(uvw[1]*nrmuvw,n)
    ww = np.tile(uvw[2]*nrmuvw,n)

    cosdl = vecarraydot([xx,yy,zz],[dd,ee,ff])
    cosmu = vecarraydot([xx,yy,zz],[uu,vv,ww])
    cosal = vecarraydot([xx,yy,zz],[aa,bb,cc])
    
    denom = 1.0/(1.0+np.absolute(cosal))
    
    xproj =  cosmu * denom
    yproj = -cosdl * denom

    hemis = np.tile('S',n)    
    if np.array(n)>1.0:
        hemis[cosal<0.0] = 'N'
    else:
        if cosal<0:
            hemis = np.tile('N',n)
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def stereotransstandard(xyz):
    '''
    Perform the standard stereographic transform on cartesian vectors
    following De Graef & McHenry
    
    Returns projected x, projected y, and the hemisphere of the projection
    '''
    import numpy as np
    x = vecarrayconvert(xyz[0])
    y = vecarrayconvert(xyz[1])
    z = vecarrayconvert(xyz[2])
    nrm = 1.0/vecarraynorm([x,y,z])
    denom = nrm/(1.0+abs(z*nrm))
    xproj =  x * denom
    yproj = -y * denom

    n = np.shape(x)
    hemis = np.tile('S',n)
    if n[0]>1.0:
        hemis[z<0.0] = 'N'
    else:
        if z<0.0:
            hemis = 'N'
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def revstereotrans(xyh,center=[0,0,1],south=[0,1,0],east=[1,0,0]):
    '''
    Reverse non-standard stereographic projection
    '''
    import numpy as np
    import numpy.linalg as npla
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed)

    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    n = np.shape(xproj)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    
    aa = abc[0]*nrmabc
    bb = abc[1]*nrmabc
    cc = abc[2]*nrmabc
    dd = fed[0]*nrmdef
    ee = fed[1]*nrmdef
    ff = fed[2]*nrmdef
    uu = uvw[0]*nrmuvw
    vv = uvw[1]*nrmuvw
    ww = uvw[2]*nrmuvw   

    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    
    cosal = (1.0 - R) / (1.0 + R)
    denom = abs(cosal) + 1.0
    cosmu = xproj * denom
    cosdl = yproj * denom
    
    # For each case we determine xyz by solving a system of linear equations
    for i in range(0,np.shape(cosal)[0]):
        xyzcoef = np.squeeze(np.array([[dd,ee,ff],[uu,vv,ww],[aa,bb,cc]]))
        xyzsoln = np.array([cosdl[i], cosmu[i], cosal[i]])
        xyz = npla.solve(xyzcoef, xyzsoln)
        x[i]= xyz[0]
        y[i]=-xyz[1]
        z[i]= xyz[2]
    
    # Correct z for the hemisphere
    z = z * m
    
    return x,y,z

#-------------------------------------------------------------------------------

def revstereotransstandard(xyh):
    '''
    Performs the reverse stereographic transform on the projected x and y
    coordinates measured from the stereographic projection.
    Reverse of De Graef and McHenry procedure.
    
    Returns the cartesian x, y, and z values of the normalized vector.
    '''
    import numpy as np
    
    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    z = (m - m * R) / (1.0 + R)
    s = m * z + 1.0
    x =  xproj * s
    y = -yproj * s

    return x,y,z

#-------------------------------------------------------------------------------

def eatrans(xyz,center=[0,0,1],south=[0,1,0],east=[1,0,0]):
    '''
    Perform non-standard equal area projections.
    
    Reference:
    [1] Kosel TH, J Mater Sci 19 (1984)
    '''
    import numpy as np
    
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed) 
    
    xyz = vecarrayconvert(xyz)
    nrmxyz = 1.0/vecarraynorm(xyz)
    
    xx = xyz[0]*nrmxyz
    yy = xyz[1]*nrmxyz
    zz = xyz[2]*nrmxyz

    n = np.shape(xx)     
    
    aa = np.tile(abc[0]*nrmabc,n)
    bb = np.tile(abc[1]*nrmabc,n)
    cc = np.tile(abc[2]*nrmabc,n)
    dd = np.tile(fed[0]*nrmdef,n)
    ee = np.tile(fed[1]*nrmdef,n)
    ff = np.tile(fed[2]*nrmdef,n)
    uu = np.tile(uvw[0]*nrmuvw,n)
    vv = np.tile(uvw[1]*nrmuvw,n)
    ww = np.tile(uvw[2]*nrmuvw,n)

    cosdl = vecarraydot([xx,yy,zz],[dd,ee,ff])
    cosmu = vecarraydot([xx,yy,zz],[uu,vv,ww])
    cosal = vecarraydot([xx,yy,zz],[aa,bb,cc])
    
    denom = 1.0/np.sqrt(1.0+np.absolute(cosal))
    
    xproj =  cosmu * denom
    yproj = -cosdl * denom

    hemis = np.tile('S',n)    
    if np.array(n)>1.0:
        hemis[cosal<0.0] = 'N'
    else:
        if cosal<0:
            hemis = np.tile('N',n)
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def eatransstandard(xyz):
    '''
    Perform the standard stereographic transform on cartesian vectors
    following De Graef & McHenry
    
    Returns projected x, projected y, and the hemisphere of the projection
    '''
    import numpy as np
    
    x=vecarrayconvert(xyz[0])
    y=vecarrayconvert(xyz[1])
    z=vecarrayconvert(xyz[2])
    nrm=1.0/vecarraynorm([x,y,z])
    denom=nrm/np.sqrt(1.0+abs(z*nrm))
    xproj =  x * denom
    yproj = -y * denom

    n=np.shape(x)
    hemis = np.tile('S',n)
    if n[0]>1.0:
        hemis[z<0.0] = 'N'
    else:
        if z<0.0:
            hemis = 'N'
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def reveatrans(xyh,center=[0,0,1],south=[0,1,0],east=[1,0,0]):
    '''
    Reverse non-standard stereographic projection
    '''
    import numpy as np
    import numpy.linalg as npla
    
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed)

    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    n = np.shape(xproj)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    
    aa = abc[0]*nrmabc
    bb = abc[1]*nrmabc
    cc = abc[2]*nrmabc
    dd = fed[0]*nrmdef
    ee = fed[1]*nrmdef
    ff = fed[2]*nrmdef
    uu = uvw[0]*nrmuvw
    vv = uvw[1]*nrmuvw
    ww = uvw[2]*nrmuvw   

    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    
    cosal = (1.0 - R) / (1.0 + R)
    denom = (np.absolute(cosal) + 1.0)**2.0
    cosmu = xproj * denom
    cosdl = yproj * denom
    
    # For each case we determine xyz by solving a system of linear equations
    for i in range(0, np.shape(cosal)[0]):
        xyzcoef = np.squeeze(np.array([[dd,ee,ff],[uu,vv,ww],[aa,bb,cc]]))
        xyzsoln = np.array([cosdl[i], cosmu[i], cosal[i]])
        xyz = npla.solve(xyzcoef, xyzsoln)
        x[i]= xyz[0]
        y[i]=-xyz[1]
        z[i]= xyz[2]
    
    # Correct z for the hemisphere
    z = z * m
    
    return x,y,z

#-------------------------------------------------------------------------------

def reveatransstandard(xyh):
    '''
    Performs the reverse stereographic transform on the projected x and y
    coordinates measured from the stereographic projection.
    
    Reverse of De Graef and McHenry procedure.
    
    Returns the cartesian x, y, and z values of the normalized vector.
    '''
    import numpy as np
    
    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    z = (m - m * R) / (1.0 + R)
    s = (m * z + 1.0)**2.0
    x =  xproj * s
    y = -yproj * s

    return x,y,z

#-------------------------------------------------------------------------------

def gnomonictrans(xyz,center=[0,0,1],south=[0,1,0],east=[1,0,0]):
    '''
    Perform the gnomonic projection, allowing different projection views
    '''
    import numpy as np
    
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed) 
    
    xyz = vecarrayconvert(xyz)
    nrmxyz = 1.0/vecarraynorm(xyz)
    
    xx = xyz[0]*nrmxyz
    yy = xyz[1]*nrmxyz
    zz = xyz[2]*nrmxyz

    n = np.shape(xx)     
    
    aa = np.tile(abc[0]*nrmabc,n)
    bb = np.tile(abc[1]*nrmabc,n)
    cc = np.tile(abc[2]*nrmabc,n)
    dd = np.tile(fed[0]*nrmdef,n)
    ee = np.tile(fed[1]*nrmdef,n)
    ff = np.tile(fed[2]*nrmdef,n)
    uu = np.tile(uvw[0]*nrmuvw,n)
    vv = np.tile(uvw[1]*nrmuvw,n)
    ww = np.tile(uvw[2]*nrmuvw,n)

    cosdl = vecarraydot([xx,yy,zz],[dd,ee,ff])
    cosmu = vecarraydot([xx,yy,zz],[uu,vv,ww])
    cosal = vecarraydot([xx,yy,zz],[aa,bb,cc])
    
    denom = 1.0/np.absolute(cosal)
    
    xproj =  cosmu * denom
    yproj = -cosdl * denom

    hemis = np.tile('S',n)    
    if np.array(n)>1.0:
        hemis[cosal<0.0] = 'N'
    else:
        if cosal<0:
            hemis = np.tile('N',n)
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def gnomonictransstandard(xyz):
    '''
    Perform the standard gnomonic projection
    
    Returns projected x, projected y, and the hemisphere of the projection
    '''    
    import numpy as np
    
    x=vecarrayconvert(xyz[0])
    y=vecarrayconvert(xyz[1])
    z=vecarrayconvert(xyz[2])
    denom=1.0/np.absolute(z)
    xproj =  x * denom
    yproj = -y * denom

    n=np.shape(x)
    hemis = np.tile('S',n)
    if n[0]>1.0:
        hemis[z<0.0] = 'N'
    else:
        if z<0.0:
            hemis = 'N'
    
    return xproj, yproj, hemis

#-------------------------------------------------------------------------------

def revgnomonictrans(xyh,center=[0,0,1],south=[0,1,0],east=[1,0,0]):
    '''
    reverse gnomonic projection allowing different projection views
    '''
    import numpy as np
    import numpy.linalg as npla    
    
    abc = vecarrayconvert(center)
    nrmabc = 1.0/vecarraynorm(abc)    
    
    uvw = vecarrayconvert(east)
    nrmuvw = 1.0/vecarraynorm(uvw)
    
    fed = vecarrayconvert(south) # 'def' is reserved in python
    nrmdef = 1.0/vecarraynorm(fed)

    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    n = np.shape(xproj)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    
    aa = abc[0]*nrmabc
    bb = abc[1]*nrmabc
    cc = abc[2]*nrmabc
    dd = fed[0]*nrmdef
    ee = fed[1]*nrmdef
    ff = fed[2]*nrmdef
    uu = uvw[0]*nrmuvw
    vv = uvw[1]*nrmuvw
    ww = uvw[2]*nrmuvw   

    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    
    cosal = (1.0 - R) / (1.0 + R)
    denom = np.absolute(cosal)
    cosmu = xproj * denom
    cosdl = yproj * denom
    
    # For each case we determine xyz by solving a system of linear equations
    for i in range(0,np.shape(cosal)[0]):
        xyzcoef = np.squeeze(np.array([[dd,ee,ff],[uu,vv,ww],[aa,bb,cc]]))
        xyzsoln = np.array([cosdl[i], cosmu[i], cosal[i]])
        xyz = npla.solve(xyzcoef, xyzsoln)
        x[i]= xyz[0]
        y[i]=-xyz[1]
        z[i]= xyz[2]
    
    # Correct z for the hemisphere
    z = z * m
    
    return x,y,z

#-------------------------------------------------------------------------------

def revgnomonictransstandard(xyh):
    '''
    performs the reverse standard gnomonic projection
    '''
    import numpy as np
    
    xproj = vecarrayconvert(xyh[0])
    yproj = vecarrayconvert(xyh[1])
    hemis = np.array(np.squeeze(xyh[2]))
    
    R = xproj*xproj + yproj*yproj
    
    m = np.ones(np.shape(hemis))
    m[hemis=='N'] = -1.0
    z = (m - m * R) / (1.0 + R)
    s =  m * z
    x =  xproj * s
    y = -yproj * s

    return x,y,z

#-------------------------------------------------------------------------------

def xtaldot(p1=1,p2=0,p3=0,
            g11=1,g12=0,g13=0,g21=0,g22=1,g23=0,g31=0,g32=0,g33=1,
            q1=1,q2=0,q3=0):
    ''' implements the dot product within the crystallographic reference frame:
        p*G\q where p and q are vectors and G is a metric matrix '''   
                
    p1  = vecarrayconvert(p1)
    p2  = vecarrayconvert(p2)
    p3  = vecarrayconvert(p3)
    q1  = vecarrayconvert(q1)
    q2  = vecarrayconvert(q2)
    q3  = vecarrayconvert(q3)
    g11 = vecarrayconvert(g11)
    g12 = vecarrayconvert(g12)
    g13 = vecarrayconvert(g13)
    g21 = vecarrayconvert(g21)
    g22 = vecarrayconvert(g22)
    g23 = vecarrayconvert(g23)
    g31 = vecarrayconvert(g31)
    g32 = vecarrayconvert(g32)
    g33 = vecarrayconvert(g33)
    
    return (g11*q1+g12*q2+g13*q3)*p1+ \
         (g21*q1+g22*q2+g23*q3)*p2+ \
         (g31*q1+g32*q2+g33*q3)*p3    

#-------------------------------------------------------------------------------

def xtalangle(p1=1,p2=0,p3=0,q1=1,q2=0,q3=0,
            g11=1,g12=0,g13=0,g21=0,g22=1,g23=0,g31=0,g32=0,g33=1):
    ''' compute the angle between two directions in a crystal'''            

    import numpy as np

    p1  = vecarrayconvert(p1)
    p2  = vecarrayconvert(p2)
    p3  = vecarrayconvert(p3)
    q1  = vecarrayconvert(q1)
    q2  = vecarrayconvert(q2)
    q3  = vecarrayconvert(q3)
    g11 = vecarrayconvert(g11)
    g12 = vecarrayconvert(g12)
    g13 = vecarrayconvert(g13)
    g21 = vecarrayconvert(g21)
    g22 = vecarrayconvert(g22)
    g23 = vecarrayconvert(g23)
    g31 = vecarrayconvert(g31)
    g32 = vecarrayconvert(g32)
    g33 = vecarrayconvert(g33)
            
    nrm=1.0/(vecarraynorm([p1,p2,p3])*vecarraynorm([q1,q2,q3]))
    
    return np.arccos(nrm * 
                xtaldot(p1,p2,p3,g11,g12,g13,g21,g22,g23,g31,g32,g33,q1,q2,q3))
    
#-------------------------------------------------------------------------------

def rationalize(v,maxval=9):
    ''' produce rational indices from fractions '''
    import numpy as np
    import copy

    v0 = vecarrayconvert(v[0])
    v1 = vecarrayconvert(v[1])
    v2 = vecarrayconvert(v[2])
    nrm=1.0/vecarraynorm([v0,v1,v2])
    v0 = v0 * nrm
    v1 = v1 * nrm
    v2 = v2 * nrm
    
    if np.shape(v0)==np.shape(v1)==np.shape(v2):
        if np.size(v0)==1:
            v0=np.array([v0])
            v1=np.array([v1])
            v2=np.array([v2])
    
        n=np.size(v[0])
        vi=np.zeros([n,maxval])
        vj=copy.copy(vi); vk=copy.copy(vi); vq=copy.copy(vi)
        for i in range(1,maxval+1):
            vx = np.around(np.float64(i) * v0)
            vy = np.around(np.float64(i) * v1)
            vz = np.around(np.float64(i) * v2)
            tmpx = np.absolute(vx); tmpy = np.absolute(vy); tmpz = np.absolute(vz)
            
            # calculate the greatest common divisor between tmpx, tmpy, and tmpz
            # we have to do this manually because fractions.gcd doesn't work on 
            # arrays and numpy doesn't yet have a gcd function implemented
            div = 1.0/vecarraygcd(vecarraygcd(tmpx,tmpy),tmpz)
          
            # multipy the irrational indices by the greatest common divisor
            vi[:,i-1] = vx * div
            vj[:,i-1] = vy * div
            vk[:,i-1] = vz * div
            nrm=1.0/vecarraynorm([vi[:,i-1],vj[:,i-1],vk[:,i-1]])
            vq[:,i-1] = np.arccos(np.amin([
                          np.tile(1.0,n),
                          vecarraydot([ vi[:,i-1]*nrm,vj[:,i-1]*nrm,vk[:,i-1]*nrm],
                                   [v[0],v[1],v[2]])
                                     ],axis=0))
        
        # extract the best match rational values
        loc=np.argmin(vq.T, axis=0)
        vi=vi[range(0, n),loc]
        vj=vj[range(0, n),loc]
        vk=vk[range(0, n),loc]
        vq=vq[range(0, n),loc]
        
        return vi, vj, vk,vq
        
    else:
        print "rationalize error:"+ \
                "check that the lengths of arguments are equal."
        
#-------------------------------------------------------------------------------

def degsymbol():
    '''returns the degree symbol for plotting'''
    return unichr(176).encode("latin-1")

#-------------------------------------------------------------------------------

def degrees(a):
    '''converts radians to degrees'''
    import numpy as np
    
    a=vecarrayconvert(a)
    return a*180.0/np.pi

#-------------------------------------------------------------------------------

def radians(a):
    '''converts degrees to radians'''
    import numpy as np
    
    a=vecarrayconvert(a)
    return a*np.pi/180.0

#------------------------------------------------------------------------------

def uniquerows(aa):
    ''' returns the number of unique rows in an array.
    
    Parameters
    ----------
    aa : numpy array
    
    Returns
    -------
    cc : numpy array
        Rows in aa without repetitions
    ia : numpy Bool array
        Indexing array such that cc = aa[ia]
    ic : numpy index array
        Indexing array such that aa = cc[ic]        
    
    Notes
    -----
    Mimics behavior of matlab function 'unique' with optional parameter 'rows'.
    Algorithm modified from a stack overflow posting [1]_.    
    
    References
    ---------
    .. [1] http://stackoverflow.com/questions/8560440/, Accessed 2012-09-10
    '''
    import numpy as np
    
    ind = np.lexsort(np.fliplr(aa).T) # indices for aa sorted by rows
    rev = np.argsort(ind) # reverse of the sorting indices
    ab = aa[ind] # ab is sorted version of aa
    dd = np.diff(ab, axis=0) # get differences between the rows
    ui = np.ones(np.shape(ab)[0], 'bool') # preallocate boolean array
    ui[1:] = (dd != 0).any(axis=1) # if difference is zero, row is not unique

    ia = ui[rev] # positions of unique rows in aa (original, unsorted array)
    cc = aa[ia] # unique rows in aa in original order

    loc = np.cumsum(np.uint64(ui))-1 # cumulative sum := locs of repeats in ab    

    # - rev[ia] gives us the indices of the unique rows in aa     
    # - argsort(rev[ia]) gives us the indices of the corresponding rows in cc 
    # - argsort(rev[ia])[loc] gives us the indexing relationship for ab from cc
    # - np.argsort(rev[ia])[loc][rev] give indexing reln in original order
    ic = np.argsort(rev[ia])[loc][rev]

    return cc, ia, ic

#------------------------------------------------------------------------------

def sortrows(a):
    '''
    sorts an array by rows
    
    Notes
    -----
    Mimics the behavior of matlab's function of the same name.
    '''
    import numpy as np
    order = np.lexsort(np.fliplr(a).T)
    return a[order]

#------------------------------------------------------------------------------

def sigdec(a, n=1):
    '''
    Rounds the elements of a to n decimals.   
    A slight modification of Peter J. Acklam's Matlab function FIXDIG      
    '''
    import numpy as np
    a = vecarrayconvert(a)
    n = vecarrayconvert(n)
    f = np.array(10.**n)
    s = np.sign(a)
    
    return s * np.around(np.absolute(a) * f) / f
    
#-------------------------------------------------------------------------------
def tic():
    ''' Timing function (like tic/toc in matlab). See also: toc'''
    import time
    return time.time()

def toc(t):
    ''' Timing function (like tic/toc in matlab). See also: tic'''
    import time
    t2 = time.time()
    print str(t2 - t) + ' seconds elapsed.'
    return t2 - t

#-------------------------------------------------------------------------------
class progbar:
    ''' console progress indicator
    
    Parameters
    ----------        
    finalcount : int
        the last number in the range
    period : float
        the amount of time that is ignored for updates to reduce slowdowns
        caused by frequent calls to the function
    message : string
        the message to be displayed above the bar in the console 
    
    Returns
    -------
    None  

    Notes
    -----        
    This is a modification of the console progress indicator class recipe by
    Larry Bates [1]_ to include ideas from Ben Mitch's matlab progbar [2]_.
    
    References
    ----------
    .. [1] L. Bates, "Console progress indicator class." Accessed 2013-02-13.
           URL: http://code.activestate.com/recipes/299207-console-text-progress-indicator-class/
    .. [2] B. Mitch, "progbar.m: general purpose progress bar for matlab." 
           Accessed 2013-02-13. 
           URL: http://www.mathworks.com/matlabcentral/fileexchange/869
    
    Examples
    --------     
    >>> # Create the progbar object
    >>> pb = progbar(finalcount=10, period=0.2, message='Progress indicator')
    >>> for i in range(1,11,1):
    ...    pb.update(i) # pass the progress values to the update method
    >>> pb.update(-1)   # update a negative value to signal process is complete
                        # and trigger the output of the elapsed time.
    

    '''       
    def __init__(self, finalcount=1, period=0.2, 
                 progresschar=None, message=''):
        import time
        import sys
        import numpy as np
        
        self.finalcount = finalcount
        self.blockcount = 0
        
        # See if caller passed me a character to use on the
        # progress bar (like "*").  If not use the block
        # character that makes it look like a real progress
        # bar.
        if not progresschar: self.block = chr(149)
        else:                self.block = progresschar

        # Get pointer to sys.stdout so I can use the write/flush
        # methods to display the progress bar.
        self.f = sys.stdout
        
        # Store the current time and minimum period between updates
        self.ctim   = time.time()
        self.itim   = time.time() # initial time
        self.period = period
        self.finished = False

        # If the final count is zero, don't start the progress gauge
        if not self.finalcount: return
        if len(message) < 50:
            val = max(np.floor(0.5 * (50 - len(message)))-1, 0)
        else:
            val = 0
        self.f.write('\n'+' '*val+message+'\n')
        self.f.write('|_____________  P R O G R E S S  ________________|\n')
        self.f.write('|----1----2----3----4----5----6----7----8----9---|' + \
                                                                 ' FINISH!\n')
        return

    def update(self, count):
        import time
        import numpy as np
       
        # Check if the elapsed time is greater than the period. If not, return.
        etim = time.time()
        if (etim - self.ctim < self.period and count != -1): return

        # If the elapsed time is greater than the period, update the bar
        self.ctim = etim        
        
        # Make sure I don't try to go off the end (e.g. >100%)
        if count == -1: count = self.finalcount
        count=min(count, self.finalcount)

        # If finalcount is zero, I don't start
        if self.finalcount:
            percentcomplete=int(np.around(100*count/self.finalcount))
            if percentcomplete < 1: percentcomplete = 1
        else:
            percentcomplete = 100
            
        blockcount = int(percentcomplete/2)
        if blockcount > self.blockcount:
            for i in range(self.blockcount,blockcount):
                self.f.write(self.block)
                self.f.flush()
                
        if percentcomplete == 100 or count == self.finalcount:
            if self.finished == False:
                self.f.write(" ELAPSED TIME: " + \
                              time.strftime('%H:%M:%S', \
                              time.gmtime(etim-self.itim)) + '\n\n')
                self.finished = True
                
        self.blockcount = blockcount
        return
        
#-------------------------------------------------------------------------------  

def iseven(x):
    ''' is it an even number?

    Returns True if x is even and False if x is odd
    
    Parameters
    ----------
    x : int
    
    Returns
    -------
    b : boolean
    '''
    import numpy as np
    
    # using np.absolute in case input is array
    xa = vecarrayconvert(x)
    b = np.mod(xa, 1) == 0 and np.absolute(np.mod(xa, 2)) <= np.spacing(1)
    return b

#------------------------------------------------------------------------------

def fileconvert_eurodecimals_to_international(filepath, outputfilepath=None, \
                                              inplace=False):
    ''' convert commas to points in numbers in text data files
    
    Converts a text file where the decimals are given by commas into one
    where the decimals are given by decimal points.
    
    Parameters
    ----------
    filepath : string
        the path to the file
    outputfilepath : string (optional)
        The path where the new file should be saved. The default is the same
        directory where the original file was stored.
    inplace : Boolean
        Flag for whether original file should be overwritten (Default=False)
    
    Notes
    -----    
    If inplace is true, it will overwrite your file! You might want to have
    a backup somewhere in this case.
    '''
    import time

    if outputfilepath==None:
        outputfilepath = \
                     time.strftime('_%Y%m%d_%H%M%S.').join(filepath.split('.'))
    
    if inplace==True:
        outputfilepath = filepath
    
    with open(filepath,'r') as f:

        newf = open(outputfilepath,'r+')

        for line in f:
            ll = line.split()
            newll = []
            for item in ll:
                if item[0].isdigit() and item[-1].isdigit() and item.find(','):
                    newll.append(item.replace(',','.'))
                else:
                    newll.append(item)
            newf.write('\t'.join(newll)+'\n')
    newf.close()

#------------------------------------------------------------------------------

def fileconvert_international_to_eurodecimals(filepath, outputfilepath=None, \
                                              inplace=False):
    ''' convert points to commas in numbers in text data files
    
    Converts a text file where the decimals are given by decimal points into
    one where the decimals are given by commas.
    
    Parameters
    ----------
    filepath : string
        the path to the file
    outputfilepath : string (optional)
        The path where the new file should be saved. The default is the same
        directory where the original file was stored.
    inplace : Boolean
        Flag for whether original file should be overwritten (Default=False)
    
    Notes
    -----    
    If inplace is true, it will overwrite your file! You might want to have
    a backup somewhere in this case.
    '''
    import time

    if outputfilepath==None:
        outputfilepath = \
                     time.strftime('_%Y%m%d_%H%M%S.').join(filepath.split('.'))
    
    if inplace==True:
        outputfilepath = filepath
    
    with open(filepath, 'r') as f:

        newf = open(outputfilepath, 'r+')

        for line in f:
            ll = line.split()
            newll = []
            for item in ll:
                if item[0].isdigit() and item[-1].isdigit() and item.find(','):
                    newll.append(item.replace('.',','))
                else:
                    newll.append(item)
            newf.write('\t'.join(newll)+'\n')
    newf.close()

#------------------------------------------------------------------------------

def polar(x=0, y=0, z=None):
    ''' polar coordinates from cartesian coordinates
       
    Converts cartesian coordinates to polar coordinates using a right-handed
    coordinate system that aligns with the standard definitions of polar 
    coordinates in 2D, i.e. theta is the azimuthal angle measured from the
    positive x axis, r is the radius of the circle. This definition is extended
    to the 3D case such that r corresponds to the radius of the sphere for 3D
    data and the additional parameter rho is the polar angle measured from the
    positive z direction. This allows the 3D spherical coordinates to be
    projected in the x-y plane as polar coordinates with r_2D = r_3D*cos(rho).
    
    Parameters
    ----------
    x : numpy array
    y : numpy array
    z : numpy array (optional)
    
    Returns
    -------
    theta : numpy array
    r     : numpy array
    rho   : numpy array (variable argument out)
        Rho is returned only when z values are passed as input parameters
    
    Notes
    -----
    The conventions used here differ from those generally used by physicists
    for spherical coordinates, which are specified in the ISO standard 
    31-11 [1]_; however, the conventions used here are consistent with the
    standard conventions used in polar coordinates (2D).
    
    References
    ----------
    [1] "Spherical coordinate system." Wikipedia. Accessed 2013-03-27.
    '''
    import numpy as np
    
    theta = np.arctan2(y, x)
    loc = np.array(theta < 0.0) * np.array(theta > 2.0*np.pi)
    theta[loc] = np.mod(theta[loc], 2.0*np.pi)
    
    loc = np.absolute(theta) < np.sqrt(np.spacing(1))
    theta[loc] = 0.0
    
    if z==None:
        r = np.sqrt(x**2.0 + y**2.0)
        return theta, r
    else:
        r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
        rho = np.arccos(z / r)
        loc = np.array(rho < 0.0) * np.array(rho > 2.0*np.pi)
        rho[loc] = np.mod(rho[loc], 2.0*np.pi)

        loc = np.absolute(rho) < np.sqrt(np.spacing(1))
        rho[loc] = 0.0
        
        return theta, r, rho

#------------------------------------------------------------------------------

def cart(theta=0, r=0, rho=None):
    ''' converts polar coordinates to cartesian coordinates in 2D

    Notes
    ----- 
    uses a right-handed 3D coordinate system that aligns with the standard
    definitions of polar coordinates in 2D, i.e. theta is the azimuthal angle
    measured from the positive x axis and rho is the polar angle measured from
    the positive z direction.
    ''' 
    import numpy as np    
    
    if rho==None:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    else:        
        x = r * np.sin(rho) * np.cos(theta)
        y = r * np.sin(rho) * np.sin(theta)
        z = r * np.cos(rho)
        return x, y, z

#------------------------------------------------------------------------------

def rand_vonMisesFisherM(n, kappa=0, mu=[1.0, 0.0, 0.0, 0.0]):
    """ random number generation from von Mises-Fisher matrix distribution
    
    Return n samples of random unit directions centered around mu with
    dispersion parameter kappa
    
    Parameters
    ----------
    n : int
        number of samples
    
    kappa : float > 0
        concentration parameter
    
    mu : iterable of floats
        central vector; length determines dimensionality m
    
    Returns
    -------
    x : n x m numpy array
        rows correspond to random unit vectors from distribution
    
    Notes
    -----
    This is a python translation of Sungkyu Jung's matlab code published on
    his website, version dated 3 Feb 2010 [1]_. It uses the modified Ulrich's
    algorithm from Wood [2]_.
    
    References
    ----------
    .. [1] S. Jung, M. Foskey, J.S. Marron, "Principal Arc Analysis on Direct
           Product Manifolds," Annnals of Applied Statistics (2011).
    .. [2] A.T.A. Wood, "Simulation of the von Mises Fisher distribution," Commun.
           Statist. 23 (1994).
    """
    import numpy as np
    import scipy.stats as stats

    # Convert mu to a 2d numpy array
    mu = np.atleast_2d(np.squeeze(np.asarray(mu).ravel()))
    
    # the dimensionality is always given by the size of mu
    m = mu.size
    
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa**2.0 + (m - 1.0)**2.0)) / (m - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (m - 1.0)*np.log(1.0 - x0**2.0)
    
    # steps 1 & 2 from [2]
    nnow = n
    ww = []
    while True:
        ntrial = np.amax([np.around(nnow * 1.2), nnow + 10.0])
        z = stats.beta.rvs((m - 1.0) / 2.0, (m - 1.0) / 2.0, size=ntrial)
        u = np.random.rand(ntrial)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        
        indicator = kappa * w + (m - 1.0) * np.log(1.0 - x0 * w) - c >= np.log(u)
        if np.sum(indicator) >= nnow:
            w1 = w[indicator]
            ww = np.hstack([ww, w1[0:nnow]])
            break
        else:
            ww = np.hstack([ww, w[indicator]])
            nnow = nnow - np.sum[indicator]
    
    # step 3 from [2]: generate n uniformly distributed m dimensional random 
    # directions, using the logic: "directions of normal distribution are
    # uniform on the sphere."
    v = np.zeros([m - 1, n])
    nr = stats.norm.rvs(1.0, size=[m - 1, n])
    for i in range(0, n):
        while True:
            ni = np.dot(nr[:, i], nr[:, i]) # length of ith vector
            # exclude too small values to avoid numerical discretization
            if ni < np.sqrt(np.spacing(np.float64(1))):
                # repeat randomization
                nr[:, i] = stats.norm.rvs(1.0, size=[m - 1, 1])
            else:
                v[:, i] = nr[:, i] / np.sqrt(ni)
                break
    
    x = np.vstack([np.tile(np.sqrt(1.0 - ww**2.0), [m - 1, 1]) * v,
                   np.atleast_2d(ww)])
    
    
    # Get the rotation matrix that rotates the data to be centered at mu   
    d = mu.size
    a = np.zeros(d)
    a[-1] = 1.0
    a = np.atleast_2d(a)
    ab = np.dot(a, mu.T)
    alpha = np.arccos(ab)
    ii = np.eye(d)
    
    if   np.abs(ab - 1) < 1e-15:
        rot =  ii
    elif np.abs(ab + 1) < 1e-15:
        rot = -ii
    else:
        c = mu - a * ab
        c = c / np.linalg.norm(c)
        aa = np.dot(a.T, c) - np.dot(c.T, a)
        rot = ii + np.sin(alpha)*aa + (np.cos(alpha) - 1.0)*(np.dot(a.T, a) + np.dot(c.T, c))
    
    return np.dot(rot.T, x).T

#------------------------------------------------------------------------------

def h5disp(f, quiet=False):
    """
    Displays the groups, datasets, and attributes in an h5 file similar to the same command in Matlab
    
    Parameters
    ----------
    f :: h5py file class
    quiet :: True/False (False default), determines whether written to console
    
    Returns
    -------
    out :: text output string (optional)
    
    Example
    -------
    >>> import h5py
    >>> f = h5py.File('my_file.h5')
    >>> out = h5disp(f)
    
    """
    import h5py

    out = '\n\nHDF5 file: {0}\n----\n'.format(f.filename)
    
    mylist = []
    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            mylist.append(name)
    f.visititems(func) # get dataset and group names
    
    oldparent = ''
    tabs = ''
    rootGroupUsed = 0
    for item in mylist:
        newparent = f[item].parent.name
        if newparent == '/':
            tabs = ''
        else:
            tabs = '    ' * (len(f[mylist[0]].name.split('/')) - 1)
        
        if newparent != oldparent:
            if newparent == '/':
                if rootGroupUsed == 0:
                    out = out + '{0}Group \'{1}\'\n'.format(tabs, newparent)
                    rootGroupUsed = 1
                oldparent = newparent
            else:
                out = out + '{0}Group \'{1}\'\n'.format(tabs, newparent)
                oldparent = newparent
                
            
        out = out + "{0}    Dataset \'{1}\': shape={2}\n".format(\
                              tabs, f[item].name.split('/')[-1], f[item].shape)

        x = h5py.AttributeManager(f[item]).items()
        if len(x) != 0:
            out = out + "{0}        Attributes:\n".format(tabs)
            for a in x:                
                out = out+ "{0}            \'{1}\': {2}\n".format(\
                                                              tabs, a[0], a[1])
    if quiet==False:   
        print out
        
    return out

#------------------------------------------------------------------------------
    
def ismember(A, B):
    import numpy as np
    return [ np.sum(a == B) for a in A ]

#------------------------------------------------------------------------------
    
def ind2sub( sizes, index, num_indices ):
    """
    Map a scalar index of a flat 1D array to the equivalent
    d-dimensional index
    
    Notes
    -----
    | 1  4  7 |      | 1,1  1,2  1,3 |
    | 2  5  8 |  --> | 2,1  2,2  2,3 |
    | 3  6  9 |      | 3,1  3,2  3,3 |
    """
    import numpy as np

    denom = num_indices
    num_dims = sizes.shape[0]
    multi_index = np.empty( ( num_dims ), np.int32 )
    for i in xrange( num_dims - 1, -1, -1 ):
        denom /= sizes[i]
        multi_index[i] = index / denom
        index = index % denom
    return multi_index

#------------------------------------------------------------------------------

def sub2ind( sizes, multi_index ):
    """
    Map a d-dimensional index to the scalar index of the equivalent flat array
    
    Notes
    -----
    | 1,1  1,2  1,3 |     | 1  4  7 | 
    | 2,1  2,2  2,3 | --> | 2  5  8 |
    | 3,1  3,2  3,3 |     | 3  6  9 |      
    """
    import numpy as np
    
    num_dims = sizes.shape[0]
    index = 0
    shift = 1
    for i in np.arange( num_dims ):
        index += shift * multi_index[i]
        shift *= sizes[i]
    return index

#------------------------------------------------------------------------------

def ssign(x):
    """ returns the sign for each item in an array    
    """
    import numpy as np
    
    y = np.ones(np.shape(x))
    y[x<0] = -1
    return y

#------------------------------------------------------------------------------

def isnegligible(x, factor=10.0):
    """ returns the indices of an array that are effectively zero
    
    Parameters
    ----------
    x : numpy array
    factor : float
        factor over float spacing to consider effectively zero
    
    Returns
    -------
    numpy bool array
    """
    import numpy as np
    
    epsilon = np.spacing(1, dtype=x.dtype)
    return np.abs(x) <= factor * epsilon

#------------------------------------------------------------------------------

def fullfile(x):
    """ builds full filename from parts

    Parameters
    ---------    
    x : list of strings
        path parts, folders, and filename
    
    Returns
    -------
    string
    
    Notes
    -----
    Provides same functionality as Matlab's FULLFILE function
    
    Example
    -------
    >>> import os
    >>> print(os.getcwd()) # print the current working directory
    >>> x = os.getcwd().split(os.sep)
    >>> print(x) # show cwd as a list of parts
    >>> cwd_path = fullfile(x)
    >>> print(cwd_path) # assemble back together into full path
    """    
    import os
    return os.sep.join(x)

#------------------------------------------------------------------------------

class helper(object):
    """ Provides system-specific information to cryspy    
    """    
    def __init__(self):
        
        import tempfile, os, platform, warnings, inspect

        # get platform-dependent temporary directory
        tmp_path = tempfile.gettempdir() 

        # Determine the path to cryspy
        cryspy_path = os.sep.join(inspect.getfile(inspect.currentframe()).split(os.sep)[0:-2])
        
        # Get the bits used by the platform
        bits, linkage = platform.architecture()
        bits = bits[0:2]
        
        # determine the operating system platform
        plat = platform.system().lower()
        
        # parse the platform bits and OS
        result = []
        extension = ''
        if plat == 'windows':
            result = 'win' + bits
            extension = '.exe'
        elif plat == 'darwin' and bits == '64':
            result = 'maci' + bits
        elif plat == 'darwin' and bits == '32':
            result = 'maci'
        elif (plat == 'linux' or plat == 'linux2') and bits == '32':
            result = 'glnx86'
        elif (plat == 'linux' or plat == 'linux2') and bits == '64':
            result = 'glnxa64'
        else:
            warnings.warn('Required c binary not compiled for your system.')
        
        # Create the path to our platform-dependent binary executables
        prgpth = fullfile([cryspy_path, 'c', 'bin', result])
        
        # Store this info in our helper object
        self.tmpdir = tmp_path
        self.arch   = result
        self.ext    = extension
        self.prgpth = prgpth
