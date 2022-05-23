# -*- coding: utf-8 -*-
'''
ovlib.orrl: Orientation relationship characterization
===============================================================================
'''

def namedOR(name):
    """ Returns ksi values for named orientation relationships
    
    Parameters
    ----------
    name : {'ks', 'nw', 'bain'}
        Orientation relationship name 
    
    Returns
    -------
    ksi_values : 1x3 numpy array of floats
        ksi values in degrees
    
    Notes
    -----
    TODO: Add plane parallel Greninger-Troiano, Kelly, etc.
    TODO: Allow 'Kurdjumov-Sachs' as well as 'ks', etc.
    """
    import numpy as np

    if isinstance(name, str):

        if name.lower() == 'ks':

            s6 = np.sqrt(6.0)
            s3 = np.sqrt(3.0)
            ksi1 = np.arccos((s6 + 1.0) / (2.0 * s3))
            ksi2 = np.arccos((s6 + 18.0) / (12.0 * s3))
            ksi3 = np.arccos((s6 + 12.0) / (6.0 * s6))
            ksi = np.array([ksi1, ksi2, ksi3])
            del s6, s3, ksi1, ksi2, ksi3

        elif name.lower() == 'nw':

            s6 = np.sqrt(6)
            s2 = np.sqrt(2)
            ksi0 = np.arccos((s2 + 1.0) / s6)
            ksi = np.array([0.0, ksi0, ksi0])

        elif name.lower() == 'bain':
  
            ksi = np.array([0.0, 0.0, 0.0])

        else:

            print 'namedOR: Unrecognized named OR'

    else:

        print 'namedOR requires a string input. Returning Bain.'
        ksi = np.array([0.0, 0.0, 0.0])
    
    return ksi * 180.0/np.pi

def getRepresentativeVariant(ksi_values):
    """ Returns quaternion of representative variant
    
    Parameters
    ----------
    ksi_values : 
    
    Returns
    -------
    psiq : quat class    
    """
    import numpy as np
    from cryspy.rot import quat

    if isinstance(ksi_values, str):
        orrl = namedOR(ksi_values)
    else:
        # Convert OR specification into radians
        orrl = ksi_values * np.pi / 180.0        
    
    csum = np.cos(orrl[0])+np.cos(orrl[1])+np.cos(orrl[2])
    x0 = 0.5 * np.sqrt(csum + 1.0)
    x = 0.5 * np.sqrt(2. * np.cos(orrl) - csum + 1.0)
          
    ksiq = quat(x0, x[0], x[1], x[2])
    
    # Calculate quaternion for normalised Bain correspondence.
    
    qw = 0.5 * np.sqrt(2.0 + np.sqrt(2.0))
    qz = (2.0 / np.sqrt(2.0)) / (4.0 * qw)
    gammaq = quat(qw, 0.0, 0.0, qz);
    
    psiq = ksiq * gammaq
    
    return psiq

def generateVariants(ksi_values):
    """ Generate variants from Kurdjumov-Sachs angles

    Returns matrices of an orientation relationship specified in Kurjumov-Sachs
    angles.

    Parameters
    ----------
    ksi_values : length 3 iterable OR {'KS', 'NW', 'Bain'}

    Returns
    -------
    vv : rmat object
        rotation matrices corresponding to variants
        
    """
    import numpy as np
    from cryspy.rot import rmat
    from cryspy.util import vecarraynorm, uniquerows, sigdec
    
    if isinstance(ksi_values, str):
    
        ksi = namedOR(ksi_values)
             
    # convert ksi radians to rotation matrices
    
    mb = np.zeros([2, 9]) 
    
    mb[0, 0] = np.cos(ksi[0])
    mb[0, 4] = np.cos(ksi[1])
    mb[0, 8] = np.cos(ksi[2])
    
    costh = 0.5 * (np.sum(np.cos(ksi)) - 1.0) # sum(cos(ksi)) is the matrix trace
    mosth = 1.0 - costh
    sinth = np.sqrt(1.0 - costh**2.0)
    
    
    r1 = np.sqrt((mb[0, 0] - costh) / mosth)
    r2 = np.sqrt((mb[0, 4] - costh) / mosth)
    r3 = np.sqrt((mb[0, 8] - costh) / mosth)
    del costh
    
    r1r2 = r1 * r2 * mosth
    r1r3 = r1 * r3 * mosth
    r2r3 = r2 * r3 * mosth
    r3st = r3 * sinth
    r2st = r2 * sinth
    r1st = r1 * sinth
    del r1, r2, r3, mosth, sinth
    
    mb[0, 5] = r2r3 - r1st
    mb[0, 7] = r2r3 + r1st
    mb[1, :] = mb[0, :]
    
    mb[0, 1] = -r1r2 + r3st
    mb[0, 2] = -r1r3 - r2st
    mb[0, 3] = -r1r2 - r3st
    mb[0, 6] = -r1r3 + r2st
    del r1r2, r1r3, r2r3, r3st, r2st, r1st
    
    mb[1, 1] = -mb[0, 1]
    mb[1, 2] = -mb[0, 2]
    mb[1, 3] = -mb[0, 3]
    mb[1, 6] = -mb[0, 6]
    # mb[0] is the 'positive' solution; mb[1] is the 'negative' solution
    
    # create Bain correspondence matrices
    bb = np.zeros([12, 9])
    bb[ 0, :] = [ 1.0, -1.0,  0.0,  1.0,  1.0,  0.0,  0.0,  0.0,  1.0]
    bb[ 1, :] = [ 0.0,  1.0, -1.0,  0.0,  1.0,  1.0,  1.0,  0.0,  0.0]
    bb[ 2, :] = [-1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0]
    bb[ 3, :] = [ 0.0,  1.0,  1.0,  0.0, -1.0,  1.0,  1.0,  0.0,  0.0]
    bb[ 4, :] = [-1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  0.0,  0.0,  1.0]
    bb[ 5, :] = [ 1.0,  0.0, -1.0,  1.0,  0.0,  1.0,  0.0, -1.0,  0.0]
    bb[ 6, :] = [ 1.0,  1.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  1.0]
    bb[ 7, :] = [-1.0,  0.0, -1.0, -1.0,  0.0,  1.0,  0.0,  1.0,  0.0]
    bb[ 8, :] = [ 0.0, -1.0,  1.0,  0.0,  1.0,  1.0, -1.0,  0.0,  0.0]
    bb[ 9, :] = [ 1.0,  0.0,  1.0,  1.0,  0.0, -1.0,  0.0,  1.0,  0.0]		
    bb[10, :] = [ 0.0, -1.0, -1.0,  0.0,  1.0, -1.0,  1.0,  0.0,  0.0]
    bb[11, :] = [-1.0,  1.0,  0.0,  1.0,  1.0,  0.0,  0.0,  0.0, -1.0]
    
    # normalize correspondence matrices
    bb = rmat.from_array(bb / vecarraynorm(bb))
    mb = rmat.from_array(mb)
    
    # produce variants
    vv = np.zeros([24, 9])
    tmp = mb[0] * bb
    vv[np.arange(0, 24, 2), :] = tmp.to_array()
    tmp = mb[1] * bb
    vv[np.arange(1, 24, 2), :] = tmp.to_array()
    
    # reduce redundancies, if they exist (as they do, for example, in NW)
    vv, ia, ic = uniquerows(sigdec(vv, 7))
    del ia, ic
    
    return rmat.from_array(vv)
    
        

#------------------------------------------------------------------------------
  
def _bg_ksi1(bincenters):
    """ returns ksi1 background
    
    Parameters
    ----------
    bincenters : numpy array
        centers of bins for ksi1 background
    
    Returns
    -------
    bg : numpy array
        background number fractions for ksi1
        
    Notes
    -----
    Background is determined by cubic spline interpolation from a large
    numerical simulation containing 17156424 ksi values measured from
    random orientations.
    
    """
    import numpy as np
    import scipy.interpolate as interp
    
    d = np.array([ [  0.00000000e+00,   0.00000000e+00],
                   [  1.50000000e-01,   8.04400000e+03],
                   [  4.50000000e-01,   2.45720000e+04],
                   [  7.50000000e-01,   4.06200000e+04],
                   [  1.05000000e+00,   5.61680000e+04],
                   [  1.35000000e+00,   7.13330000e+04],
                   [  1.65000000e+00,   8.80860000e+04],
                   [  1.95000000e+00,   1.02050000e+05],
                   [  2.25000000e+00,   1.17010000e+05],
                   [  2.55000000e+00,   1.31760000e+05],
                   [  2.85000000e+00,   1.47720000e+05],
                   [  3.15000000e+00,   1.60180000e+05],
                   [  3.45000000e+00,   1.75440000e+05],
                   [  3.75000000e+00,   1.88830000e+05],
                   [  4.05000000e+00,   2.03510000e+05],
                   [  4.35000000e+00,   2.16520000e+05],
                   [  4.65000000e+00,   2.29790000e+05],
                   [  4.95000000e+00,   2.43770000e+05],
                   [  5.25000000e+00,   2.56080000e+05],
                   [  5.55000000e+00,   2.68910000e+05],
                   [  5.85000000e+00,   2.81760000e+05],
                   [  6.15000000e+00,   2.93690000e+05],
                   [  6.45000000e+00,   3.06460000e+05],
                   [  6.75000000e+00,   3.16310000e+05],
                   [  7.05000000e+00,   3.27680000e+05],
                   [  7.35000000e+00,   3.40790000e+05],
                   [  7.65000000e+00,   3.50150000e+05],
                   [  7.95000000e+00,   3.63730000e+05],
                   [  8.25000000e+00,   3.74700000e+05],
                   [  8.55000000e+00,   3.83250000e+05],
                   [  8.85000000e+00,   3.93360000e+05],
                   [  9.15000000e+00,   4.01330000e+05],
                   [  9.45000000e+00,   4.08800000e+05],
                   [  9.75000000e+00,   4.10170000e+05],
                   [  1.00500000e+01,   4.16810000e+05],
                   [  1.03500000e+01,   4.22100000e+05],
                   [  1.06500000e+01,   4.24130000e+05],
                   [  1.09500000e+01,   4.25640000e+05],
                   [  1.12500000e+01,   4.26720000e+05],
                   [  1.15500000e+01,   4.26860000e+05],
                   [  1.18500000e+01,   4.25850000e+05],
                   [  1.21500000e+01,   4.25010000e+05],
                   [  1.24500000e+01,   4.19670000e+05],
                   [  1.27500000e+01,   4.14030000e+05],
                   [  1.30500000e+01,   4.06510000e+05],
                   [  1.33500000e+01,   3.97900000e+05],
                   [  1.36500000e+01,   3.85480000e+05],
                   [  1.39500000e+01,   3.72070000e+05],
                   [  1.42500000e+01,   3.55970000e+05],
                   [  1.45500000e+01,   3.36740000e+05],
                   [  1.48500000e+01,   3.11690000e+05],
                   [  1.51500000e+01,   2.79820000e+05],
                   [  1.54500000e+01,   2.57130000e+05],
                   [  1.57500000e+01,   2.35370000e+05],
                   [  1.60500000e+01,   2.18230000e+05],
                   [  1.63500000e+01,   2.01140000e+05],
                   [  1.66500000e+01,   1.84260000e+05],
                   [  1.69500000e+01,   1.68320000e+05],
                   [  1.72500000e+01,   1.54590000e+05],
                   [  1.75500000e+01,   1.39760000e+05],
                   [  1.78500000e+01,   1.27400000e+05],
                   [  1.81500000e+01,   1.14020000e+05],
                   [  1.84500000e+01,   1.03180000e+05],
                   [  1.87500000e+01,   9.14750000e+04],
                   [  1.90500000e+01,   8.00980000e+04],
                   [  1.93500000e+01,   7.07470000e+04],
                   [  1.96500000e+01,   6.10580000e+04],
                   [  1.99500000e+01,   5.08750000e+04],
                   [  2.02500000e+01,   4.25510000e+04],
                   [  2.05500000e+01,   3.39610000e+04],
                   [  2.08500000e+01,   2.52270000e+04],
                   [  2.11500000e+01,   1.73040000e+04],
                   [  2.14500000e+01,   1.12930000e+04],
                   [  2.17500000e+01,   7.17200000e+03],
                   [  2.20500000e+01,   3.76000000e+03],
                   [  2.23500000e+01,   1.55200000e+03],
                   [  2.26500000e+01,   3.94000000e+02],
                   [  2.29500000e+01,   8.00000000e+00],
                   [  2.32500000e+01,   0.00000000e+00],
                   [  2.35500000e+01,   0.00000000e+00],
                   [  2.38500000e+01,   0.00000000e+00],
                   [  2.41500000e+01,   0.00000000e+00],
                   [  2.44500000e+01,   0.00000000e+00],
                   [  2.47500000e+01,   0.00000000e+00],
                   [  2.50500000e+01,   0.00000000e+00],
                   [  2.53500000e+01,   0.00000000e+00],
                   [  2.56500000e+01,   0.00000000e+00],
                   [  2.59500000e+01,   0.00000000e+00],
                   [  2.62500000e+01,   0.00000000e+00],
                   [  2.65500000e+01,   0.00000000e+00],
                   [  2.68500000e+01,   0.00000000e+00],
                   [  2.71500000e+01,   0.00000000e+00],
                   [  2.74500000e+01,   0.00000000e+00],
                   [  2.77500000e+01,   0.00000000e+00],
                   [  2.80500000e+01,   0.00000000e+00],
                   [  2.83500000e+01,   0.00000000e+00],
                   [  2.86500000e+01,   0.00000000e+00],
                   [  2.89500000e+01,   0.00000000e+00],
                   [  2.92500000e+01,   0.00000000e+00],
                   [  2.95500000e+01,   0.00000000e+00],
                   [  2.98500000e+01,   0.00000000e+00],
                   [  3.01500000e+01,   0.00000000e+00],
                   [  3.04500000e+01,   0.00000000e+00],
                   [  3.07500000e+01,   0.00000000e+00],
                   [  3.10500000e+01,   0.00000000e+00],
                   [  3.13500000e+01,   0.00000000e+00],
                   [  3.16500000e+01,   0.00000000e+00],
                   [  3.19500000e+01,   0.00000000e+00],
                   [  3.22500000e+01,   0.00000000e+00],
                   [  3.25500000e+01,   0.00000000e+00],
                   [  3.28500000e+01,   0.00000000e+00],
                   [  3.31500000e+01,   0.00000000e+00],
                   [  3.34500000e+01,   0.00000000e+00],
                   [  3.37500000e+01,   0.00000000e+00],
                   [  3.40500000e+01,   0.00000000e+00],
                   [  3.43500000e+01,   0.00000000e+00],
                   [  3.46500000e+01,   0.00000000e+00]])

    minval = np.sqrt(np.spacing(np.float32(1)))
    cj = interp.UnivariateSpline(d[:,0],
                                 d[:,1]+minval, 
                                 k=1, s=0.5)
    
    # Interpolate the spline for our bin array into a background for our histogram
    bg = cj(bincenters)
    bg[bg < minval] = minval # do not allow division by zero
    
    return bg

def _bg_ksi2(bincenters):
    """ returns ksi1 background
    
    Parameters
    ----------
    bincenters : numpy array
        centers of bins for ksi1 background
    
    Returns
    -------
    bg : numpy array
        background number fractions for ksi1
        
    Notes
    -----
    Background is determined by cubic spline interpolation from a large
    numerical simulation containing 17156424 ksi values measured from
    random orientations. The bin spacing in this histogram is 0.3 degrees.
    
    """
    import numpy as np
    import scipy.interpolate as interp
    
    d = np.array([ [  0.00000000e+00,   0.00000000e+00],
                   [  1.50000000e-01,   3.80000000e+01],
                   [  4.50000000e-01,   2.53000000e+02],
                   [  7.50000000e-01,   6.37000000e+02],
                   [  1.05000000e+00,   1.30400000e+03],
                   [  1.35000000e+00,   2.14800000e+03],
                   [  1.65000000e+00,   3.15500000e+03],
                   [  1.95000000e+00,   4.51300000e+03],
                   [  2.25000000e+00,   6.03800000e+03],
                   [  2.55000000e+00,   7.55500000e+03],
                   [  2.85000000e+00,   9.49000000e+03],
                   [  3.15000000e+00,   1.15950000e+04],
                   [  3.45000000e+00,   1.37350000e+04],
                   [  3.75000000e+00,   1.64950000e+04],
                   [  4.05000000e+00,   1.88700000e+04],
                   [  4.35000000e+00,   2.18850000e+04],
                   [  4.65000000e+00,   2.54230000e+04],
                   [  4.95000000e+00,   2.88190000e+04],
                   [  5.25000000e+00,   3.21320000e+04],
                   [  5.55000000e+00,   3.60380000e+04],
                   [  5.85000000e+00,   3.99840000e+04],
                   [  6.15000000e+00,   4.36950000e+04],
                   [  6.45000000e+00,   4.81920000e+04],
                   [  6.75000000e+00,   5.32350000e+04],
                   [  7.05000000e+00,   5.78030000e+04],
                   [  7.35000000e+00,   6.26690000e+04],
                   [  7.65000000e+00,   6.77040000e+04],
                   [  7.95000000e+00,   7.24380000e+04],
                   [  8.25000000e+00,   7.90320000e+04],
                   [  8.55000000e+00,   8.46900000e+04],
                   [  8.85000000e+00,   9.05220000e+04],
                   [  9.15000000e+00,   9.71920000e+04],
                   [  9.45000000e+00,   1.03820000e+05],
                   [  9.75000000e+00,   1.10760000e+05],
                   [  1.00500000e+01,   1.17170000e+05],
                   [  1.03500000e+01,   1.23890000e+05],
                   [  1.06500000e+01,   1.31260000e+05],
                   [  1.09500000e+01,   1.38800000e+05],
                   [  1.12500000e+01,   1.47670000e+05],
                   [  1.15500000e+01,   1.54370000e+05],
                   [  1.18500000e+01,   1.62540000e+05],
                   [  1.21500000e+01,   1.71450000e+05],
                   [  1.24500000e+01,   1.80180000e+05],
                   [  1.27500000e+01,   1.88610000e+05],
                   [  1.30500000e+01,   1.96360000e+05],
                   [  1.33500000e+01,   2.06070000e+05],
                   [  1.36500000e+01,   2.15760000e+05],
                   [  1.39500000e+01,   2.25380000e+05],
                   [  1.42500000e+01,   2.34630000e+05],
                   [  1.45500000e+01,   2.45320000e+05],
                   [  1.48500000e+01,   2.56620000e+05],
                   [  1.51500000e+01,   2.62000000e+05],
                   [  1.54500000e+01,   2.64870000e+05],
                   [  1.57500000e+01,   2.69010000e+05],
                   [  1.60500000e+01,   2.71560000e+05],
                   [  1.63500000e+01,   2.74630000e+05],
                   [  1.66500000e+01,   2.78040000e+05],
                   [  1.69500000e+01,   2.80570000e+05],
                   [  1.72500000e+01,   2.82790000e+05],
                   [  1.75500000e+01,   2.85960000e+05],
                   [  1.78500000e+01,   2.88360000e+05],
                   [  1.81500000e+01,   2.89220000e+05],
                   [  1.84500000e+01,   2.92980000e+05],
                   [  1.87500000e+01,   2.93430000e+05],
                   [  1.90500000e+01,   2.96900000e+05],
                   [  1.93500000e+01,   2.97160000e+05],
                   [  1.96500000e+01,   2.98050000e+05],
                   [  1.99500000e+01,   3.00280000e+05],
                   [  2.02500000e+01,   3.02770000e+05],
                   [  2.05500000e+01,   3.01800000e+05],
                   [  2.08500000e+01,   3.04190000e+05],
                   [  2.11500000e+01,   3.05480000e+05],
                   [  2.14500000e+01,   3.02770000e+05],
                   [  2.17500000e+01,   3.01660000e+05],
                   [  2.20500000e+01,   2.96840000e+05],
                   [  2.23500000e+01,   2.91940000e+05],
                   [  2.26500000e+01,   2.87030000e+05],
                   [  2.29500000e+01,   2.79490000e+05],
                   [  2.32500000e+01,   2.74410000e+05],
                   [  2.35500000e+01,   2.68910000e+05],
                   [  2.38500000e+01,   2.60700000e+05],
                   [  2.41500000e+01,   2.54640000e+05],
                   [  2.44500000e+01,   2.47580000e+05],
                   [  2.47500000e+01,   2.42160000e+05],
                   [  2.50500000e+01,   2.36140000e+05],
                   [  2.53500000e+01,   2.29720000e+05],
                   [  2.56500000e+01,   2.26630000e+05],
                   [  2.59500000e+01,   2.17650000e+05],
                   [  2.62500000e+01,   2.12590000e+05],
                   [  2.65500000e+01,   2.06500000e+05],
                   [  2.68500000e+01,   2.01920000e+05],
                   [  2.71500000e+01,   1.95560000e+05],
                   [  2.74500000e+01,   1.88290000e+05],
                   [  2.77500000e+01,   1.84200000e+05],
                   [  2.80500000e+01,   1.78390000e+05],
                   [  2.83500000e+01,   1.72670000e+05],
                   [  2.86500000e+01,   1.66430000e+05],
                   [  2.89500000e+01,   1.61550000e+05],
                   [  2.92500000e+01,   1.55230000e+05],
                   [  2.95500000e+01,   1.50620000e+05],
                   [  2.98500000e+01,   1.44690000e+05],
                   [  3.01500000e+01,   1.00740000e+05],
                   [  3.04500000e+01,   6.55290000e+04],
                   [  3.07500000e+01,   4.09580000e+04],
                   [  3.10500000e+01,   2.05860000e+04],
                   [  3.13500000e+01,   3.76200000e+03],
                   [  3.16500000e+01,   0.00000000e+00],
                   [  3.19500000e+01,   0.00000000e+00],
                   [  3.22500000e+01,   0.00000000e+00],
                   [  3.25500000e+01,   0.00000000e+00],
                   [  3.28500000e+01,   0.00000000e+00],
                   [  3.31500000e+01,   0.00000000e+00],
                   [  3.34500000e+01,   0.00000000e+00],
                   [  3.37500000e+01,   0.00000000e+00],
                   [  3.40500000e+01,   0.00000000e+00],
                   [  3.43500000e+01,   0.00000000e+00],
                   [  3.46500000e+01,   0.00000000e+00]])

    minval = np.sqrt(np.spacing(np.float32(1)))
    cj = interp.UnivariateSpline(d[:,0],
                                 d[:,1]+minval, 
                                 k=1, s=0.5)
    
    # Interpolate the spline for our bin array into a background for our histogram
    bg = cj(bincenters)
    bg[bg < minval] = minval # do not allow division by zero
    
    return bg    

def _bg_ksi3(bincenters):
    """ returns ksi1 background
    
    Parameters
    ----------
    bincenters : numpy array
        centers of bins for ksi1 background
    
    Returns
    -------
    bg : numpy array
        background number fractions for ksi1
        
    Notes
    -----
    Background is determined by cubic spline interpolation from a large
    numerical simulation containing 17156424 ksi values measured from
    random orientations. The bin spacing in this histogram is 0.3 degrees.
    
    """
    import numpy as np
    import scipy.interpolate as interp
    
    d = np.array([ [  0.00000000e+00,   0.00000000e+00],
                   [  1.50000000e-01,   2.02400000e+03],
                   [  4.50000000e-01,   6.40000000e+03],
                   [  7.50000000e-01,   1.04980000e+04],
                   [  1.05000000e+00,   1.44530000e+04],
                   [  1.35000000e+00,   1.87920000e+04],
                   [  1.65000000e+00,   2.27030000e+04],
                   [  1.95000000e+00,   2.67350000e+04],
                   [  2.25000000e+00,   3.07190000e+04],
                   [  2.55000000e+00,   3.50120000e+04],
                   [  2.85000000e+00,   3.91140000e+04],
                   [  3.15000000e+00,   4.28890000e+04],
                   [  3.45000000e+00,   4.71630000e+04],
                   [  3.75000000e+00,   5.13210000e+04],
                   [  4.05000000e+00,   5.43970000e+04],
                   [  4.35000000e+00,   5.88480000e+04],
                   [  4.65000000e+00,   6.29620000e+04],
                   [  4.95000000e+00,   6.75770000e+04],
                   [  5.25000000e+00,   7.14270000e+04],
                   [  5.55000000e+00,   7.46270000e+04],
                   [  5.85000000e+00,   7.91680000e+04],
                   [  6.15000000e+00,   8.24090000e+04],
                   [  6.45000000e+00,   8.65320000e+04],
                   [  6.75000000e+00,   9.06090000e+04],
                   [  7.05000000e+00,   9.47120000e+04],
                   [  7.35000000e+00,   9.79920000e+04],
                   [  7.65000000e+00,   1.01590000e+05],
                   [  7.95000000e+00,   1.05900000e+05],
                   [  8.25000000e+00,   1.09020000e+05],
                   [  8.55000000e+00,   1.12940000e+05],
                   [  8.85000000e+00,   1.17490000e+05],
                   [  9.15000000e+00,   1.20630000e+05],
                   [  9.45000000e+00,   1.23850000e+05],
                   [  9.75000000e+00,   1.27110000e+05],
                   [  1.00500000e+01,   1.33170000e+05],
                   [  1.03500000e+01,   1.35750000e+05],
                   [  1.06500000e+01,   1.38000000e+05],
                   [  1.09500000e+01,   1.42120000e+05],
                   [  1.12500000e+01,   1.45870000e+05],
                   [  1.15500000e+01,   1.49250000e+05],
                   [  1.18500000e+01,   1.52450000e+05],
                   [  1.21500000e+01,   1.56420000e+05],
                   [  1.24500000e+01,   1.59460000e+05],
                   [  1.27500000e+01,   1.63300000e+05],
                   [  1.30500000e+01,   1.65540000e+05],
                   [  1.33500000e+01,   1.68700000e+05],
                   [  1.36500000e+01,   1.71120000e+05],
                   [  1.39500000e+01,   1.75060000e+05],
                   [  1.42500000e+01,   1.78070000e+05],
                   [  1.45500000e+01,   1.80540000e+05],
                   [  1.48500000e+01,   1.83390000e+05],
                   [  1.51500000e+01,   1.85630000e+05],
                   [  1.54500000e+01,   1.89110000e+05],
                   [  1.57500000e+01,   1.91660000e+05],
                   [  1.60500000e+01,   1.95150000e+05],
                   [  1.63500000e+01,   1.97860000e+05],
                   [  1.66500000e+01,   1.99910000e+05],
                   [  1.69500000e+01,   2.03460000e+05],
                   [  1.72500000e+01,   2.05680000e+05],
                   [  1.75500000e+01,   2.08830000e+05],
                   [  1.78500000e+01,   2.10760000e+05],
                   [  1.81500000e+01,   2.12770000e+05],
                   [  1.84500000e+01,   2.14450000e+05],
                   [  1.87500000e+01,   2.17540000e+05],
                   [  1.90500000e+01,   2.19990000e+05],
                   [  1.93500000e+01,   2.22920000e+05],
                   [  1.96500000e+01,   2.24490000e+05],
                   [  1.99500000e+01,   2.25390000e+05],
                   [  2.02500000e+01,   2.27330000e+05],
                   [  2.05500000e+01,   2.29780000e+05],
                   [  2.08500000e+01,   2.31760000e+05],
                   [  2.11500000e+01,   2.34130000e+05],
                   [  2.14500000e+01,   2.34930000e+05],
                   [  2.17500000e+01,   2.36170000e+05],
                   [  2.20500000e+01,   2.39060000e+05],
                   [  2.23500000e+01,   2.39200000e+05],
                   [  2.26500000e+01,   2.40280000e+05],
                   [  2.29500000e+01,   2.41970000e+05],
                   [  2.32500000e+01,   2.42410000e+05],
                   [  2.35500000e+01,   2.44470000e+05],
                   [  2.38500000e+01,   2.44630000e+05],
                   [  2.41500000e+01,   2.45470000e+05],
                   [  2.44500000e+01,   2.46940000e+05],
                   [  2.47500000e+01,   2.46660000e+05],
                   [  2.50500000e+01,   2.48020000e+05],
                   [  2.53500000e+01,   2.49590000e+05],
                   [  2.56500000e+01,   2.50060000e+05],
                   [  2.59500000e+01,   2.49730000e+05],
                   [  2.62500000e+01,   2.49300000e+05],
                   [  2.65500000e+01,   2.49770000e+05],
                   [  2.68500000e+01,   2.51230000e+05],
                   [  2.71500000e+01,   2.50670000e+05],
                   [  2.74500000e+01,   2.50110000e+05],
                   [  2.77500000e+01,   2.53140000e+05],
                   [  2.80500000e+01,   2.50100000e+05],
                   [  2.83500000e+01,   2.49790000e+05],
                   [  2.86500000e+01,   2.49210000e+05],
                   [  2.89500000e+01,   2.48700000e+05],
                   [  2.92500000e+01,   2.47540000e+05],
                   [  2.95500000e+01,   2.47380000e+05],
                   [  2.98500000e+01,   2.47910000e+05],
                   [  3.01500000e+01,   1.88290000e+05],
                   [  3.04500000e+01,   1.38640000e+05],
                   [  3.07500000e+01,   1.07030000e+05],
                   [  3.10500000e+01,   8.20110000e+04],
                   [  3.13500000e+01,   5.92320000e+04],
                   [  3.16500000e+01,   4.13020000e+04],
                   [  3.19500000e+01,   2.75660000e+04],
                   [  3.22500000e+01,   1.75590000e+04],
                   [  3.25500000e+01,   9.84800000e+03],
                   [  3.28500000e+01,   4.43200000e+03],
                   [  3.31500000e+01,   1.59800000e+03],
                   [  3.34500000e+01,   1.18000000e+02],
                   [  3.37500000e+01,   0.00000000e+00],
                   [  3.40500000e+01,   0.00000000e+00],
                   [  3.43500000e+01,   0.00000000e+00],
                   [  3.46500000e+01,   0.00000000e+00]])

    minval = np.sqrt(np.spacing(np.float32(1)))
    cj = interp.UnivariateSpline(d[:,0],
                                 d[:,1]+minval, 
                                 k=1, s=0.5)
    
    # Interpolate the spline for our bin array into a background for our histogram
    bg = cj(bincenters)
    bg[bg < minval] = minval # do not allow division by zero
    
    return bg       

def ksi1hist(ksi1values, binwidth=0.5, return_everything=False):
    """ histogram ksi1 data
    
    Parameters
    ----------
    ksi1values : numpy array of floats
        ksi1values, in degrees
    
    binwidth : numpy float (optional)
        width of bins, in degrees
    
    return_everything : bool (optional)
        flag for whether             
    
    Returns
    -------
    fcorr : numpy array
        background-corrected ksi1 number fractions
    
    bm : numpy array
        mean values for the histogram bins
    
    ba : (optional, with return_everything flag) numpy array
        histogram bin edges
    
    fraw : (optional, with return_everything flag) numpy array
        ksi1 number fractions before background correction
    """
    import numpy as np
    
    # Histogram ksi data
    ba = np.r_[0:35:binwidth]
    nb = ba.size
    bm = 0.5 * (ba[0:nb-1] + ba[1:nb])
    fraw, tmp = np.histogram(ksi1values, ba, density=True)
    fraw = fraw / np.sum(fraw) 
    
    bg = _bg_ksi1(bm)

    # Correct the histogram for measurement
    fcorr = fraw / bg
    fcorr = fcorr / np.sum(fcorr)
    
    if return_everything == False:
        return fcorr, bm
    else:
        return fcorr, bm, ba, fraw
    
def ksi2hist(ksi2values, binwidth=0.5, return_everything=False):
    """ histogram ksi2 data
    
    Parameters
    ----------
    ksi1values : numpy array of floats
        ksi1values, in degrees
    
    binwidth : numpy float (optional)
        width of bins, in degrees
    
    return_everything : bool (optional)
        flag for whether             
    
    Returns
    -------
    fcorr : numpy array
        background-corrected ksi1 number fractions
    
    bm : numpy array
        mean values for the histogram bins
    
    ba : (optional, with return_everything flag) numpy array
        histogram bin edges
    
    fraw : (optional, with return_everything flag) numpy array
        ksi1 number fractions before background correction
    """
    import numpy as np
    
    # Histogram ksi data
    ba = np.r_[0:35:binwidth]
    nb = ba.size
    bm = 0.5 * (ba[0:nb-1] + ba[1:nb])
    fraw, tmp = np.histogram(ksi2values, ba, density=True)
    fraw = fraw / np.sum(fraw) 
    
    bg = _bg_ksi2(bm)

    # Correct the histogram for measurement
    fcorr = fraw / bg
    fcorr = fcorr / np.sum(fcorr)
    
    if return_everything == False:
        return fcorr, bm
    else:
        return fcorr, bm, ba, fraw          

def ksi3hist(ksi3values, binwidth=0.5, return_everything=False):
    """ histogram ksi3 data
    
    Parameters
    ----------
    ksi3values : numpy array of floats
        ksi1values, in degrees
    
    binwidth : numpy float (optional)
        width of bins, in degrees
    
    return_everything : bool (optional)
        flag for whether             
    
    Returns
    -------
    fcorr : numpy array
        background-corrected ksi1 number fractions
    
    bm : numpy array
        mean values for the histogram bins
    
    ba : (optional, with return_everything flag) numpy array
        histogram bin edges
    
    fraw : (optional, with return_everything flag) numpy array
        ksi1 number fractions before background correction
    """
    import numpy as np
    
    # Histogram ksi data
    ba = np.r_[0:35:binwidth]
    nb = ba.size
    bm = 0.5 * (ba[0:nb-1] + ba[1:nb])
    fraw, tmp = np.histogram(ksi3values, ba, density=True)
    fraw = fraw / np.sum(fraw) 
    
    bg = _bg_ksi3(bm)

    # Correct the histogram for measurement
    fcorr = fraw / bg
    fcorr = fcorr / np.sum(fcorr)
    
    if return_everything == False:
        return fcorr, bm
    else:
        return fcorr, bm, ba, fraw

def fit_foldnorm(f, bm, return_info=False, accuracy=None, iterations=1000):
    """ fit ksi histogram to folded normal distribution
    
    Parameters
    ----------
    fcorr : co            
    
    Returns
    -------
    mu   : float
        folded normal fit location parameter
    sig  : float
        folded normal fit shape parameter
    rss  : float
        final residual sum of squares difference between the fit and
        the histogram
    info : tuple
        (1) the number of iterations required for fitting
        (2) the exit mode from the optimizer (see notes)
        (3) message describing exit mode
    
    Notes
    -----
    Minimization is performed by scipy.optimize.fmin_slsqp. Refer to
    the documentation for this function for further details on exit
    mode returns.            
    """
    import numpy as np
    from scipy.special import erf    
    import scipy.optimize as spop
    
    if accuracy == None: # default to floating point accuracy
        accuracy = np.spacing(np.float32(0))            
    
    # Estimate the distribution fitting parameters and then minimize the RMS
    # Note that RMS error is linearly related to log likelihood, so we should get
    # comparable results in both minimizations
    m1 = np.dot(bm, f)          # first moment
    m2 = np.dot(bm**2.0, f)     # second moment
    m4 = np.dot(bm**4.0, f)     # fourth moment
    s = np.dot((bm-m1)**2.0, f) # standard deviation
    x = np.arange(0.0, np.amax(bm)+np.amax(bm)/1000.0, np.amax(bm)/1000.0)
    
    # Elandt finds that the 1st and 2nd moments give a better approximation
    # when m1/s is greater than about 1.35, and the 2nd and 4th moment
    # method is better otherwise.
    if m1/s > 1.35: # estimate parameters by 1st and 2nd moments
        a  = m1**2.0 / m2 # ratio of moments
        def i0(x): # Eq 2 [2]
            return 1.0 - 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        def g(th, i0, a):
            return (np.sqrt(2.0/np.pi) * np.exp(-0.5 * th**2.0) - \
                   th * (1.0 - 2.0 * i0(-th)))**2.0 / (a * (1.0 + th**2.0)) - 1.0
        
        # Find the value of G closest to zero, limiting iterations
        # (fminbnd is not sufficiently robust in our case because G 
        # sometimes is close to zero but does not necessarily intersect the 
        # x axis in the limit of the half normal distribution). Assuming 
        # the function will only come close to zero once, then we will just
        # search for the the solution to the equation iteratively
        lb  = 0.0
        ub  = np.amax(x)
        res = np.inf
        nit = 0
        j = 0
        while (res > accuracy and
                nit < 1000 and 
                (ub - lb) / 1000.0 > accuracy):
            
            # set a search range
            qtm = np.arange(lb, ub+(ub-lb)/1000.0, (ub-lb)/1000.0)
            
            tmp  = np.absolute(g(qtm, i0, a))
            resn = np.amin(tmp) # get value closest to zero in range
            j    = np.argmin(tmp)
    
            ub = qtm[np.amin([j + 1, qtm.size - 1])] # get new upper bound
            lb = qtm[np.amax([j - 1, 0])] # get new lower bound
            nit += 1 # increment number of iterations counter
        th=qtm[j]
        del qtm, j, ub, lb, res, g, a, i0, nit, tmp
        
    else: # Estimate Parameters by second and fourth moments
    
        b = m4 / m2**2.0 # ratio of moments
        
        def h(q, b): # Eq. 22 [2]
            return b * (1.0 + q**2.0)**2.0 - (3.0 + 6.0 * q**2.0 + q**4.0) 
            
        # Find the value of h closest to zero, limiting iterations.
        # Same method as above.
        lb  = 0.0
        ub  = np.amax(x)
        res = np.inf
        nit = 0
        j   = 0
        while (res > accuracy and
                nit < 1000 and 
                np.absolute(ub-lb) / 1000.0 > accuracy):
            
            # set a search range
            qtm = np.arange(lb, ub+(ub-lb)/1000.0, (ub-lb)/1000.0) 
            
            tmp = np.absolute(h(qtm, b))
            resn = np.amin(tmp) # get value closest to zero in range
            j    = np.argmin(tmp)
            if np.isreal(resn):
                res = resn # in case min cannot be found
            ub  = qtm[np.amin([j + 1, qtm.size - 1])] # get new upper bound
            lb  = qtm[np.amax([j - 1, 0])] # get new lower bound
            nit += 1 # increment number of iterations counter

        th = qtm[j]
        del qtm, j, ub, lb, res, h, b, nit, tmp
    del m2, m4
    
    # Compute initial guess values
    sgi = np.sqrt((s**2.0 + m1**2.0) / (1.0 + th**2.0))
    mui = th * sgi
    
    # Define minimization function as a least squares problem
    def ff(params):
        import numpy as np
        import scipy.stats as stats
        fx = stats.foldnorm.pdf(bm, params[0], loc=0, scale=params[1])
        fx = fx/np.sum(fx)
        return np.sum((f-fx)**2.0)
    
    # Perform constrained minimization
    mub = (accuracy, np.inf)
    sgb = (accuracy, np.inf) 
    out, rss, its, imode, smode = spop.fmin_slsqp(ff, x0=[mui, sgi], 
                                                  bounds=[mub, sgb], 
                                                  full_output=True,
                                                  iter=iterations,
                                                  acc=accuracy,
                                                  iprint=1)
    
    mu  = out[0]
    sig = out[1]
    
    if return_info==False:
        return mu, sig
    else:
        return mu, sig, rss, (its, imode, smode)

def foldnorm_mode(mu, sig, accuracy=None):
    """ mode/peak of the folded normal distribution
    
    Parameters
    ----------
    mu : float
        location parameter
    sig : float
        shape parameter
    
    Returns
    -------
    expected_value : float
        E(Y)
    
    Notes
    -----
    Equation comes from the documentation for the folded normal distribution in
    the R statistics package.
    http://hosho.ees.hokudai.ac.jp/~kubo/Rdoc/library/VGAM/html/fnormal1.html  
    """
    import scipy.stats as stats
    import numpy as np
    import scipy.optimize as spop
    
    if accuracy == None: # default to floating point accuracy
        accuracy = np.spacing(np.float32(0))  
        
    fn = stats.foldnorm(mu, loc=0, scale=sig)
    def chk(x):
        val = -fn.pdf(x)
        return val
    
    res = spop.minimize(chk, 0.0, method='nelder-mead', 
                        options={'xtol': 1e-8, 'disp': False})
    
    return res.x[0]
           

def fit_ksivals(ksi1vals, ksi2vals, ksi3vals, binwidth=0.5, makeplot=True, \
                accuracy=None):
                    
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    f1c, bm, ba, f1r = ksi1hist(ksi1vals, 
                                 binwidth, return_everything=True)
                                 
    f2c, bm, ba, f2r = ksi2hist(ksi2vals, 
                                 binwidth, return_everything=True)
                                 
    f3c, bm, ba, f3r = ksi3hist(ksi3vals, 
                                 binwidth, return_everything=True)

    mu1, sig1, rss1, dat1 = fit_foldnorm(f1c, bm, return_info=True,
                                         accuracy=accuracy)
                                         
    mu2, sig2, rss2, dat2 = fit_foldnorm(f2c, bm, return_info=True,
                                         accuracy=accuracy)
                                         
    mu3, sig3, rss3, dat3 = fit_foldnorm(f3c, bm, return_info=True,
                                         accuracy=accuracy)
    
    print foldnorm_mode(mu1, sig1), foldnorm_mode(mu2, sig2),  foldnorm_mode(mu3, sig3) # FIXME: make output clearer to the user

    if makeplot==False:
        
        return np.array([[mu1, sig1],[mu2, sig2],[mu3, sig3]])
        
    else:

        # parameters we might want to adjust
        xlimv = [0.0, 15.0]
        ylimv = [0.0, 15.0]
        xtickstep = 3
        ytickstep = 3
        bgdistclr = [0.8,0.8,0.8]
        uncorrclr = [0.55, 0.55, 0.55]
        ksi1faceclr = [1.0, 1.0, 1.0]
        ksi2faceclr = [1.0, 1.0, 1.0]
        ksi3faceclr = [1.0, 1.0, 1.0]
        fontsz=12
        
        xv = np.arange(0.0, np.amax(bm) + np.amax(bm) / 1000.0, 
                      np.amax(bm) / 1000.0)
                      
        # Create an awesome plot
        fig = plt.figure(figsize=(15, 4), dpi=96)
        fig.patch.set_facecolor('w')        
        
        # ksi1 subplot
        ax1 = fig.add_subplot(131)
        f = stats.foldnorm.pdf(xv, mu1, scale=sig1)
        f = xv.size*f/np.sum(f)
        ax1.plot(xv, f, '-k', linewidth=1)
        ax1.bar(bm-binwidth/2.0, f1r*bm.size, width=binwidth,
                facecolor=uncorrclr, edgecolor=uncorrclr)
        ax1.bar(bm-binwidth/2.0, f1c*bm.size, width=binwidth, 
                facecolor=ksi1faceclr, edgecolor=[0.0, 0.0, 0.0])
        bg = _bg_ksi1(xv)
        bg = bg/np.sum(bg)
        ax1.fill_between(x=xv, y1=bg*bg.size, y2=0, facecolors=bgdistclr, edgecolors=bgdistclr)
        ax1.text(0.97, 0.97, 
                 r'$Mo_{\xi1}$ = '+foldnorm_mode(mu1, sig1).astype('|S5')
                 +r'$^\circ$'+'\n'+
                 r'$\mu_{\xi1}$ = '+mu1.astype('|S5')+r'$^\circ$'+'\n'+
                 r'$\sigma_{\xi1}$ = '+sig1.astype('|S5')+r'$^\circ$', 
                 fontsize=fontsz,
                 bbox=dict(edgecolor='k', facecolor='w'),
                 transform = ax1.transAxes,
                 horizontalalignment='right',
                 verticalalignment='top')
        ax1.set_xlim(xlimv)
        ax1.set_ylim(ylimv)
        ax1.set_xticks(np.arange(xlimv[0],xlimv[1]+xtickstep, xtickstep))
        ax1.set_yticks(np.arange(ylimv[0],ylimv[1]+ytickstep, ytickstep))

        # ksi2 subplot
        ax2 = fig.add_subplot(132)
        f = stats.foldnorm.pdf(xv, mu2, scale=sig2)
        f = xv.size*f/np.sum(f)
        ax2.plot(xv, f, '-k', linewidth=1)
        ax2.bar(bm-binwidth/2.0, f2r*bm.size, width=binwidth,
                facecolor=uncorrclr, edgecolor=uncorrclr)
        ax2.bar(bm-binwidth/2.0, f2c*bm.size, width=binwidth, 
                facecolor=ksi2faceclr, edgecolor=[0.0, 0.0, 0.0])
        bg = _bg_ksi2(xv)
        bg = bg/np.sum(bg)
        ax2.fill_between(x=xv, y1=bg*bg.size, y2=0, facecolors=bgdistclr, edgecolors=bgdistclr)
        ax2.text(0.03, 0.97, 
                 r'$Mo_{\xi2}$ = '+foldnorm_mode(mu2, sig2).astype('|S5')
                 +r'$^\circ$'+'\n'+
                 r'$\mu_{\xi1}$ = '+mu2.astype('|S5')+r'$^\circ$'+'\n'+
                 r'$\sigma_{\xi1}$ = '+sig2.astype('|S5')+r'$^\circ$',
                 fontsize=fontsz,
                 bbox=dict(edgecolor='k', facecolor='w'),
                 transform = ax2.transAxes,
                 horizontalalignment='left',
                 verticalalignment='top')
        ax2.set_xlim(xlimv)
        ax2.set_ylim(ylimv)
        ax2.set_xticks(np.arange(xlimv[0],xlimv[1]+xtickstep, xtickstep))
        ax2.set_yticks(np.arange(ylimv[0],ylimv[1]+ytickstep, ytickstep))


        # ksi2 subplot
        ax3 = fig.add_subplot(133)
        f = stats.foldnorm.pdf(xv, mu3, scale=sig3)
        f = xv.size*f/np.sum(f)
        ax3.plot(xv, f, '-k', linewidth=1)
        ax3.bar(bm-binwidth/2.0, f3r*bm.size, width=binwidth,
                facecolor=uncorrclr, edgecolor=uncorrclr)
        ax3.bar(bm-binwidth/2.0, f3c*bm.size, width=binwidth, 
                facecolor=ksi3faceclr, edgecolor=[0.0, 0.0, 0.0])
        bg = _bg_ksi3(xv)
        bg = bg/np.sum(bg)
        ax3.fill_between(x=xv, y1=bg*bg.size, y2=0, facecolors=bgdistclr, edgecolors=bgdistclr)
        ax3.text(0.03, 0.97, 
                 r'$Mo_{\xi3}$ = '+foldnorm_mode(mu3, sig3).astype('|S5')
                 +r'$^\circ$'+'\n'+
                 r'$\mu_{\xi1}$ = '+mu3.astype('|S5')+r'$^\circ$'+'\n'+
                 r'$\sigma_{\xi1}$ = '+sig3.astype('|S5')+r'$^\circ$', 
                 fontsize=fontsz,
                 bbox=dict(edgecolor='k', facecolor='w'),
                 transform = ax3.transAxes,
                 horizontalalignment='left',
                 verticalalignment='top')
        ax3.set_xlim(xlimv)
        ax3.set_ylim(ylimv)
        ax3.set_xticks(np.arange(xlimv[0],xlimv[1]+xtickstep, xtickstep))
        ax3.set_yticks(np.arange(ylimv[0],ylimv[1]+ytickstep, ytickstep))

        
        ax1.set_ylabel(r'Relative Number Fraction')
        ax1.set_xlabel(r'$\xi_1$ ($^\circ$)', fontsize=fontsz*1.2)
        ax2.set_xlabel(r'$\xi_2$ ($^\circ$)', fontsize=fontsz*1.2)
        ax3.set_xlabel(r'$\xi_3$ ($^\circ$)', fontsize=fontsz*1.2)
        fig.subplots_adjust(bottom=0.15)
        plt.show()
        
        return np.array([[mu1, sig1],[mu2, sig2],[mu3, sig3]]), \
                fig, ax1, ax2, ax3