def generateVariants(ksi_values):
    '''
    Generate variants from Kurdjumov-Sachs angles

    Returns matrices of an orientation relationship specified in Kurjumov-Sachs
    angles.

    
    Parameters
    ----------
    ksi_values : length 3 iterable OR {'KS', 'NW', 'Bain'}


    Returns
    -------
    vv : rmat object
        rotation matrices corresponding to variants
    '''
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



def namedOR(name):

    '''
    Returns ksi values for named orientation relationships


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
    '''
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
            print('namedOR: Unrecognized named OR')

    else:
        print('namedOR requires a string input. Returning Bain.')
        ksi = np.array([0.0, 0.0, 0.0])

    return ksi * 180.0/np.pi