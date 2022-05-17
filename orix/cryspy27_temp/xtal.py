# -*- coding: utf-8 -*-
'''
ovlib.xtal: Crystallography
===============================================================================
'''

def point_group_number_from_space_group_number(spacegroup):
    """ extract point group from space group
    
    Get the point group number from the space group number
    
    Parameters
    ----------
    spacegroup : int
        space group number
    
    Returns
    -------
    pointgroup : int
        point group number
    
    Notes
    -----
    Point groups and space groups are numbered according to the international
    notation, NOT python 0-based indexing.
    """

    pg = 1
    if spacegroup >   1: pg =  2
    if spacegroup >   2: pg =  3
    if spacegroup >   5: pg =  4
    if spacegroup >   9: pg =  5
    if spacegroup >  15: pg =  6
    if spacegroup >  24: pg =  7
    if spacegroup >  46: pg =  8
    if spacegroup >  74: pg =  9
    if spacegroup >  80: pg = 10
    if spacegroup >  82: pg = 11
    if spacegroup >  88: pg = 12
    if spacegroup >  98: pg = 13
    if spacegroup > 110: pg = 14
    if spacegroup > 122: pg = 15
    if spacegroup > 142: pg = 16
    if spacegroup > 146: pg = 17
    if spacegroup > 148: pg = 18
    if spacegroup > 155: pg = 19
    if spacegroup > 161: pg = 20
    if spacegroup > 167: pg = 21
    if spacegroup > 173: pg = 22
    if spacegroup > 174: pg = 23
    if spacegroup > 176: pg = 24
    if spacegroup > 182: pg = 25
    if spacegroup > 186: pg = 26
    if spacegroup > 190: pg = 27
    if spacegroup > 194: pg = 28
    if spacegroup > 199: pg = 29
    if spacegroup > 206: pg = 30
    if spacegroup > 214: pg = 31
    if spacegroup > 220: pg = 32

    return pg

def laueclass(pointgroup, notation='international'):
    ''' get the Laue class
    
    Returns the Laue class of the given point group.

    Parameters
    ----------
    pointgroup : int
        Point group number
    notation : str (optional)
        Name of convention used: {'schoenflies', 'geo', 'international', 'tsl'}
        Default: 'international'
    
    Returns
    -------
    laueclass : int
        the number corresponding to the Laue class of the point group
    
    Notes
    -----
    The Laue class of a point group is the next higher-order group in the same
    Bravais lattice that contains a center of symmetry.
    
    ===========  ============  ===============  =====  =========================
    Point Group  Space Groups  Bravais Lattice  Name   Notes
    ===========  ============  ===============  =====  =========================
     1           1             Triclinic         1
     2           2                              -1
     3           3-5           Monoclinic        2
     4           6-9                             m
     5           10-15                           2/m
     6           16-24         Orthorhombic      222
     7           25-46                           mm2   39: bc in MDG code (why not in 222?)
     8           47-74                           mmm
     9           75-80         Tetragonal        4
    10           81-82                          -4
    11           83-88                           4/m
    12           89-98                           422   90: bg and no c in MDG code (why not in Tetrag 4?)
    13           99-110                          4mm
    14           111-122                        -42m   115-120: bmj
    15           123-142                        4/mmm
    16           143-146       Trigonal           3
    17           147-148       (Rhombohedral)    -3
    18           149-155                          32   149,151,153: nf
    19           156-161                          3m   157,159: nl
    20           162-167                         -3m   162,163: nf
    21           168-173       Hexagonal          6
    22           174                             -6
    23           175-176                          6/m
    24           177-182                          622
    25           183-186                          6mm
    26           187-190                         -6m2  187,188: nik
    27           191-194                        6/mmm
    28           195-199       Cubic              23
    29           200-206                          m3
    30           207-214                          432
    31           215-220                         -43m
    32           221-230                         m-3m
    ===========  ============  ===============  =====  =========================
    '''

    if type(pointgroup)==str:
        pgid = interpret_point_group_name(pointgroup, notation)
    elif type(pointgroup)==int:
        pgid = pointgroup
    else:
        print 'pointgroup input to laueclass was neither a string \
               nor an integer. Using laue class m.'
        pgid = 1

    if pgid > 32:
        print 'Invalid point group number. Returning 32.'

    laue = 32
    if pgid <= 29:
        laue = 29
    if pgid <= 27:
        laue = 27
    if pgid <= 23:
        laue = 23
    if pgid <= 20:
        laue = 20
    if pgid <= 17:
        laue = 17
    if pgid <= 15:
        laue = 15
    if pgid <= 11:
        laue = 11
    if pgid <= 8:
        laue = 8
    if pgid <= 5:
        laue = 5
    if pgid <= 2:
        laue = 2

    return laue

#-------------------------------------------------------------------------------

def ctf_laue_from_laue_group(laueclass):
    ''' channel text file laue group interpretation
    
    Converts between the identification scheme for Laue groups used in ovlib
    (which is the point group corresponding to the Laue class) to the scheme
    used in Channel Text Files (*.ctf).
    
    Parameters
    ----------
    laueclass : int
        Point group number corresponding to the laue class of the phase.
    
    Returns
    -------
    ctflaue : int
        The Laue class number (as used in *.ctf files)
    '''

    # convert to internal naming scheme
    pgid = laueclass
    del laueclass

    laue = 11
    if pgid == 29:
        laue = 10
    if pgid == 27:
        laue = 9
    if pgid == 23:
        laue = 8
    if pgid == 20:
        laue = 7
    if pgid == 17:
        laue = 6
    if pgid == 15:
        laue = 5
    if pgid == 11:
        laue = 4
    if pgid == 8:
        laue = 3
    if pgid == 5:
        laue = 2
    if pgid == 2:
        laue = 1

    return laue


#-------------------------------------------------------------------------------

def rotationelements(point_group_name, notation='international'):
    """
    Function for generating rotational symmetries
    
    Parameters
    ----------
    point_group_name : str
        The name of the point group.
       
    notation : string (optional)
        Name of convention used: {'schoenflies', 'geo', 'international', 'tsl'}
        Default: 'international'
    
    Returns
    -------
    rotsymm : quaternion class
        The set of rotational symmetries.
    
    Notes
    -----
    As the name suggests, this function returns only the rotational symmetry
    elements and NOT the full symmetry, which may include rotoinversions. Thus,
    this should be used carefully and only applied to orientations. Many
    symmetry-related operations on vectors or miller indices should use the
    full set of symmetry operations.    
    """
    import numpy as np
    import numpy.linalg as npla
    import cryspy.util as util
    
    pge = pointgroupelements(point_group_name, notation)
    n = pge.numel()
    a = pge.to_array()

    # proper rotations have determinants == positive unity
    determinant = np.zeros([n,1])
    for i in range(0,n):
        determinant[i] = npla.det(a[i,:].reshape(3,3))

    determinant = util.sigdec(determinant, 1)
    
    if np.shape(determinant)[0] > 1:
        rotsymm = pge[np.squeeze(determinant == 1)]
    else:
        rotsymm = pge[0]

    return rotsymm

#-------------------------------------------------------------------------------

def interpret_point_group_number(point_group_number, notation='international'):
    """ convert point group number to point group name
    
    Parses a point group number and converts it to the point group name in the
    desired notation.

    Parameters
    ----------
    point_group_number : int {1:32}
    notation : string (optional)
        Name of convention used: {'schoenflies', 'geo', 'international', 'tsl'}
        Default: 'international'

    Returns
    -------
    point_group_name : string

    Notes
    -----
    - In the Schoenflies and geometric notations, there are multiple possible
      identification names for some of the point groups. For these cases,
      this function only returns one of the possible names.
    - The geometric notation information comes from [1]_.
    - 'tsl' is the convention used in TSL/EDAX *.ang files. It bears some
      resemblance to the geometric convention

    References
    ----------
    .. [1] D. Hestenes, J. Holt. "The Crystallographic Space Groups in
       Geometric Algebra." J Math Phy, 2007.
    """
    if notation=='international': # default
        opts=['1'    , # 1
              '-1'   , # 2
              '2'    , # 3
              'm'    , # 4
              '2/m'  , # 5
              '222'  , # 6
              'mm2'  , # 7
              'mmm'  , # 8
              '4'    , # 9
              '-4'   , #10
              '4/m'  , #11
              '422'  , #12
              '4mm'  , #13
              '42m'  , #14
              '4/mmm', #15
              '3'    , #16
              '-3'   , #17
              '32'   , #18
              '3m'   , #19
              '-3m'  , #20
              '6'    , #21
              '-6'   , #22
              '6/m'  , #23
              '622'  , #24
              '6mm'  , #25
              '-6m2' , #26
              '6/mmm', #27
              '23'   , #28
              'm-3'  , #29
              '432'  , #30
              '-43m' , #31
              'm-3m' ] #32

    elif notation=='schoenflies':
        opts=[  'C1' , # 1
                'S2' , # 2
                'C2' , # 3
                'C1h', # 4
                'C2h', # 5
                'V'  , # 6
                'C2v', # 7
                'D2h', # 8
                'C4' , # 9
                'S4' , #10
                'C4h', #11
                'D4' , #12
                'C4v', #13
                'D2d', #14
                'D4h', #15
                'C3' , #16
                'S6' , #17
                'D3' , #18
                'C3v', #19
                'D3d', #20
                'C6' , #21
                'C3h', #22
                'C6h', #23
                'D6' , #24
                'C6v', #25
                'D3h', #26
                'D6h', #27
                'T'  , #28
                'Th' , #29
                'O'  , #30
                'Td' , #31
                'Oh' ] #32

    elif notation=="geo":
        opts=[ '-1'  , # 1
               '-2-2', # 2
               '-2'  , # 3
               '1'   , # 4
               '-22' , # 5
               '-2-2', # 6
               '2'   , # 7
               '22'  , # 8
               '-4'  , # 9
               '-4-2', #10
               '-42' , #11
               '-4-2', #12
               '4'   , #13
               '4-2' , #14
             '42'    , #15
             '-3'    , #16
             '-6-2'  , #17
             '-3-2'  , #18
             '3'     , #19
             '6-2'   , #20
             '-6'    , #21
             '-32'   , #22
             '-62'   , #23
             '-6-2'  , #24
             '6'     , #25
             '32'    , #26
             '62'    , #27
             '-3-3'  , #28
             '4-3'   , #29
             '-4-3'  , #30
             '-33'   , #31
             '-43'   ] #32

    elif notation=="tsl":
    # TSL uses the rotation symmetries of the corresponding Laue group.
        opts=[   '1' , # 1
                 '1' , # 2
                 '20', # 3
                 '20', # 4
                 '20', # 5
                 '22', # 6
                 '22', # 7
                 '22', # 8
                 '4' , # 9
                 '4' , #10
                 '4' , #11
                 '42', #12
                 '42', #13
                 '42', #14
                 '42', #15
                 '3' , #16
                 '3' , #17
                 '32', #18
                 '32', #19
                 '32', #20
                 '6' , #21
                 '6' , #22
                 '6' , #23
                 '62', #24
                 '62', #25
                 '62', #26
                 '62', #27
                 '23', #28
                 '23', #29
                 '43', #30
                 '43', #31
                 '43'] #32

    return opts[point_group_number - 1]

#-------------------------------------------------------------------------------

def interpret_point_group_name(point_group_name, notation='international'):
    ''' convert point group name to number
    
    Parses a point group name and converts it to the point group number.

    Parameters
    ----------
    point_group_name : string
        Name of point group in Schoenflies, geometric, international, or TSL
        convention.
    notation : string
        Name of convention used: {'schoenflies', 'geo', 'international', 'tsl'}
        Default: 'international'

    Returns
    -------
    point_group_number : integer

    Notes
    -----
    - 'tsl' is the convention used in TSL/EDAX *.ang files. It bears some
      resemblance to the geometric convention.
    '''
    # The 32 point groups in international and schoenflies notations have
    # no conflicting names
    if ((notation==None) or
        (notation=='schoenflies') or
        (notation=='international')): # default

        # Assign our possible options
        opts=[            '1','C1',#1
                          '-1','S2','Ci',#2
                          '2','C2',#3
                          'm','C1h','Cs',#4
                          '2/m','C2h',#5
                          '222','D2','V',#6
                          'mm2','C2v',#7
                          'mmm','D2h','Vh',#8
                          '4','C4',#9
                          '-4','S4',#10
                          '4/m','C4h',#11
                          '422','D4',#12
                          '4mm','C4v',#13
                          '42m','D2d','Vd','-42m',#14
                          '4/mmm','D4h',#15
                          '3','C3',#16
                          '-3','S6','C31','C3i',#17
                          '32','D3',#18
                          '3m','C3v',#19
                          '-3m','D3d',#20
                          '6','C6',#21
                          '-6','C3h',#22
                          '6/m','C6h',#23
                          '622','D6',#24
                          '6mm','C6v',#25
                          '6m2','-6m2','-62m','62m','D3h',#26
                          '6/mmm','D6h',#27
                          '23','T',#28
                          'm3','m-3','Th',#29
                          '432','O',#30
                          '43m','-43m','Td',#31
                          'm3m','m-3m','Oh'#32
                          ]

        # assign the key to understanding those options
        key = [           0,0,
                          1,1,1,
                          2,2,
                          3,3,3,
                          4,4,
                          5,5,5,
                          6,6,
                          7,7,7,
                          8,8,
                          9,9,
                          10,10,
                          11,11,
                          12,12,
                          13,13,13,13,
                          14,14,
                          15,15,
                          16,16,16,16,
                          17,17,
                          18,18,
                          19,19,
                          20,20,
                          21,21,
                          22,22,
                          23,23,
                          24,24,
                          25,25,25,25,25,
                          26,26,
                          27,27,
                          28,28,28,
                          29,29,
                          30,30,30,
                          31,31,31];

    if notation=="geo":
    # The geometric convention conflicts terribly with the international convention
        opts=['-1',
             '-2-2',
             '-2',
             '1',
             '-22',
             '-2-2',
             '2',
             '22',
             '-4',
             '-4-2',
             '-42',
             '-4-2',
             '4',
             '4-2',
             '42',
             '-3',
             '-6-2',
             '-3-2',
             '3',
             '6-2',
             '-6',
             '-32',
             '-62',
             '-6-2',
             '6',
             '32',
             '62',
             '-3-3',
             '4-3',
             '-4-3',
             '33','-33',
             '43','-43']
        key=[0,
             1,
             2,
             3,
             4,
             5,
             6,
             7,
             8,
             9,
             10,
             11,
             12,
             13,
             14,
             15,
             16,
             17,
             18,
             19,
             20,
             21,
             22,
             23,
             24,
             25,
             26,
             27,
             28,
             29,
             30,30,
             31,31]

    if notation=="tsl":
    # TSL uses the rotation symmetries of the corresponding Laue group.
        opts=['1', #2
             '2','20', #5 (also allowing "2" since "20" does not appear to
                       #   follow the rest of the convention)
             '22', #8
             '4', #11
             '42', #15
             '3', #17
             '32', #20
             '6', #23
             '62', #27
             '23', #29
             '43'] #32
        key=[1,4,4,7,10,14,16,19,22,26,28,31]

    # Get the point group number from the point group name, converting indexing
    point_group_id_number=key[opts.index(point_group_name)] + 1

    return point_group_id_number


def pointgroupelements(point_group_name, notation='international'):
    ''' generate point group elements
    
    Function for generating point group symmetry elements containing 
    subfunctions for creating the point group symmetries.
    
    Parameters
    ----------
    point_group_name : {str, int}
        The name of the point group.

    notation : str
        Name of convention used: {'schoenflies', 'geo', 'international', 'tsl'}
        Default: 'international'    
    
    Returns
    -------
    pge : rmat class
        Rotation matrices for the point group elements.

    Notes
    -----
    This is a Python translation of the codes from [1]_.
    
    References
    ----------
    .. [1] M. De Graef, "Introduction to Conventional Transmission Electron
           Microscopy." Cambridge University Press, 2003.
    '''

    def make_generator_string(point_group_id_number):
        """
        
        Notes:
        - Need to check that point groups 39 and 90 are correct.
        - The matrices produced may be incorrect for space groups 115-120,
          149, 151, 153, 157, 159, 162, 163, 187, and 188 due to setting
          issues. This is a total of 15 out of 230 space groups.
        """

    #       Code      Point Group     Space Groups     Bravais Lattice    Name  Notes
        opts=[  'a',     #  1               1       --- Triclinic           1
                'h',     #  2               2                              -1
                'c',     #  3              3-5      --- Monoclinic          2
                'j',     #  4              6-9                              m
                'ch',    #  5             10-15                            2/m
                'bc',    #  6             16-24     --- Orthorhombic       222
                'bj',    #  7             25-46                            mm2  39: bc in MDG code (why not in 222?)
                'bch',   #  8             47-74                            mmm
                'bg',    #  9             75-80     --- Tetragonal          4
                'bm',    # 10             81-82                            -4
                'bgh',   # 11             83-88                            4/m
                'bgc',   # 12             89-98                            422  90: bg and no c in MDG code (why not in Tetrag 4?)
                'bgj',   # 13             99-110                           4mm
                'bmc',   # 14            111-122                          -42m  115-120: bmj
                'bgch',  # 15            123-142                          4/mmm
                'n',     # 16            143-146    --- Trigonal            3
                'nh',    # 17            147-148        (Rhombohedral)     -3
                'ne',    # 18            149-155                           32   149,151,153: nf
                'nk',    # 19            156-161                           3m   157,159: nl
                'neh',   # 20            162-167                          -3m   162,163: nf
                'nb',    # 21            168-173    --- Hexagonal          6
                'ni',    # 22              174                            -6
                'nbh',   # 23            175-176                          6/m
                'nbe',   # 24            177-182                          622
                'nbk',   # 25            183-186                          6mm
                'nie',   # 26            187-190                         -6m2   187,188: nik
                'nbeh',  # 27            191-194                         6/mmm
                'bcd',   # 28            195-199    --- Cubic              23
                'bcdh',  # 29            200-206                           m3
                'bcde',  # 30            207-214                          432
                'bcdl',  # 31            215-220                         -43m
                'bcdeh'  # 32            221-230                         m-3m
                ]
        # pass the point group number through the option list to obtain the
        # generator string
        generator_string=opts[point_group_id_number]

        return generator_string

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    def interpret_generator_string(generator_string):
        import numpy as np
        
        n = len(generator_string)
        generator_matrices = np.zeros([n,9])
        for i in range(0,n):
            tmp = make_generator_matrix(generator_string[i]).T
            generator_matrices[i,:] = tmp.reshape(1,9)

        return generator_matrices

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    def make_generator_matrix(t):
    # Create the generator matrices. Follows De Graef's approach.
        import numpy as np

        gmx = np.zeros([3,3]) # preallocate the generator matrix
        if t == 'a':
                gmx[0,0]= 1.0; gmx[1,1]= 1.0; gmx[2,2]= 1.0;
        elif t == 'b':
                gmx[0,0]=-1.0; gmx[1,1]=-1.0; gmx[2,2]= 1.0;
        elif t == 'c':
                gmx[0,0]=-1.0; gmx[1,1]= 1.0; gmx[2,2]=-1.0;
        elif t == 'd':
                gmx[0,2]= 1.0; gmx[1,0]= 1.0; gmx[2,1]= 1.0;
        elif t == 'e':
                gmx[0,1]= 1.0; gmx[1,0]= 1.0; gmx[2,2]=-1.0;
        elif t == 'f':
                gmx[0,1]=-1.0; gmx[1,0]=-1.0; gmx[2,2]=-1.0;
        elif t == 'g':
                gmx[0,1]=-1.0; gmx[1,0]= 1.0; gmx[2,2]= 1.0;
        elif t == 'h':
                gmx[0,0]=-1.0; gmx[1,1]=-1.0; gmx[2,2]=-1.0;
        elif t == 'i':
                gmx[0,0]= 1.0; gmx[1,1]= 1.0; gmx[2,2]=-1.0;
        elif t == 'j':
                gmx[0,0]= 1.0; gmx[1,1]=-1.0; gmx[2,2]= 1.0;
        elif t == 'k':
                gmx[0,1]=-1.0; gmx[1,0]=-1.0; gmx[2,2]= 1.0;
        elif t == 'l':
                gmx[0,1]= 1.0; gmx[1,0]= 1.0; gmx[2,2]= 1.0;
        elif t == 'm':
                gmx[0,1]= 1.0; gmx[1,0]=-1.0; gmx[2,2]=-1.0;
        elif t == 'n':
                gmx[0,1]=-1.0; gmx[1,0]= 1.0; gmx[1,1]=-1.0; gmx[2,2]= 1.0;

        return gmx

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    def generate_point_group_elements(generator_matrices):
        '''

        Multiply the generator matrices together until no new matrices are
        created. The correct number of matrices are produced (see OJ Curnow,
        Chemical Education Today 84 (2007) p1430 for a nice mnemonic trick).
        I have some concerns that the exact matrices produced in the present
        code may, however, not all be correct. See "notes" in
        make_generator_string.
        
        Parameters
        ----------
        generator_matrices : numpy array
        
        Returns
        -------
        
        '''
        import cryspy.util as util
        import numpy as np

        n1 = 0
        n2 = generator_matrices.shape[0]

        while n1<n2:
            for i in range(n1,n2):
                for j in range(0,n2):
                    tmp = np.dot(generator_matrices[i,:].reshape(3,3).T,
                                 generator_matrices[j,:].reshape(3,3).T).T

                    generator_matrices = np.concatenate((generator_matrices,
                                              tmp.reshape(1,9)), axis=0)

            generator_matrices, loc, tmp = util.uniquerows(generator_matrices)
            n1 = n2
            n2 = generator_matrices.shape[0]-1

        return generator_matrices

    ###########################################################################

    #if point_group_name.__class__
    point_group_number = interpret_point_group_name(point_group_name, notation)

    generator_string = make_generator_string(point_group_number - 1)

    generator_matrices = interpret_generator_string(generator_string)

    point_group_symmetry_matrices = generate_point_group_elements(
                                                            generator_matrices)

    import cryspy.rot as rot
    point_group_symmetry_elements = rot.rmat(
                                       g11 = point_group_symmetry_matrices[:,0],
                                       g12 = point_group_symmetry_matrices[:,1],
                                       g13 = point_group_symmetry_matrices[:,2],
                                       g21 = point_group_symmetry_matrices[:,3],
                                       g22 = point_group_symmetry_matrices[:,4],
                                       g23 = point_group_symmetry_matrices[:,5],
                                       g31 = point_group_symmetry_matrices[:,6],
                                       g32 = point_group_symmetry_matrices[:,7],
                                       g33 = point_group_symmetry_matrices[:,8],
                                         )

    return point_group_symmetry_elements

#-------------------------------------------------------------------------------

class lattsite(object):
    '''Lattice Site class

    REFERENCES:

    - M. De Graef, M. E. McHenry, "Structure of Materials: An Introduction to
      Crystallography, Diffraction, and Symmetry." New York, NY: Cambridge
      University Press, 2007.

    - B. D. Cullity, S. R. Stock, "Elements of X-Ray Diffraction." 3rd Ed. Upper
      Saddle River, NJ: Prentice Hall, 2001.

    - D. E. Sands, "Introduction to Crystallography." Mineaola, NY:
      Courier Dover Publications, 1993. p68, Eq. 3.3.
    '''

    def __init__(self, x=0, y=0, z=0):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type u=..., v=..., w=... etc.
        import numpy as np
        import cryspy.util as util

        # check that the shapes of all are the same
        if np.shape(x)==np.shape(y)==np.shape(z):

            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...
            self.x = util.vecarrayconvert(x)
            self.y = util.vecarrayconvert(y)
            self.z = util.vecarrayconvert(z)

        else:
            print "lattvec construction error: check that the lengths of u,"\
                  " v, and w are all the same."
            return None

#-------------------------------------------------------------------------------

    def rotate(self, q2):
            '''
            Multiplication of lattice site coordinates.

            When multiplied by quaternion, we rotate the coordinates of the
            lattice site
            '''
            import numpy as np
            import cryspy.rot as rot
            
            if isinstance(q2, rot.quat):
                t2 =   q2.a * q2.b
                t3 =   q2.a * q2.c
                t4 =   q2.a * q2.d
                t5 =  -q2.b * q2.b
                t6 =   q2.b * q2.c
                t7 =   q2.b * q2.d
                t8 =  -q2.c * q2.c
                t9 =   q2.c * q2.d
                t10 = -q2.d * q2.d

                v1new = np.array([2.0*( (t8 + t10)*self.x +
                                        (t6 -  t4)*self.y +
                                        (t3 + t7)*self.z ) + self.x])
                v2new = np.array([2.0*( (t4 +  t6)*self.x +
                                        (t5 + t10)*self.y +
                                        (t9 - t2)*self.z ) + self.y])
                v3new = np.array([2.0*( (t7 -  t3)*self.x +
                                        (t2 +  t9)*self.y +
                                        (t5 + t8)*self.z ) + self.z])

                return lattsite(x=v1new, y=v2new, z=v3new)

#-------------------------------------------------------------------------------

    def to_cartesian(self, unit_cell):
        '''
        multiply by direct structure matrix to get cartesian vector
        '''
        if isinstance(unit_cell, unitcell):
            
            d = unit_cell.d
            vx = d[0, 0] * self.x + d[0, 1] * self.y + d[0, 2] * self.z
            vy = d[1, 0] * self.x + d[1, 1] * self.y + d[1, 2] * self.z
            vz = d[2, 0] * self.x + d[2, 1] * self.y + d[2, 2] * self.z

            return [vx, vy, vz]

#-------------------------------------------------------------------------------

    @classmethod
    def from_cartesian(cls, arg, unit_cell, maxval=9):
        '''
        multiply by inverse transpose direct structure matrix
        '''
        import cryspy.util as util
        
        if isinstance(unit_cell, unitcell):
            x = arg[0]
            y = arg[1]
            z = arg[2]
            d = unit_cell.dinv.T
            un = d[0, 0] * x + d[0, 1] * y + d[0, 2] * z
            vn = d[1, 0] * x + d[1, 1] * y + d[1, 2] * z
            wn = d[2, 0] * x + d[2, 1] * y + d[2, 2] * z
            u, v, w, dev = util.rationalize([un, vn, wn], maxval)

            return cls(u, v, w), dev

#-------------------------------------------------------------------------------

    def distance(self, site2, unit_cell):
        '''
        returns the distance between two lattice sites.
        '''
        import numpy as np
        import cryspy.util as util
        if isinstance(site2, lattsite) and isinstance(unit_cell, unitcell):
            x = site2.x - self.x
            y = site2.y - self.y
            z = site2.z - self.z

            # multiply by metric matrix to get cartesian vector
            m = unit_cell.m
            return np.sqrt(util.xtaldot(p1=x,p2=y,p3=z,
                                g11=m[0,0], g12=m[0,1], g13=m[0,2],
                                g21=m[1,0], g22=m[1,1], g23=m[1,2],
                                g31=m[2,0], g32=m[2,1], g33=m[2,2],
                                q1=x, q2=y, q3=z))

#-------------------------------------------------------------------------------

    def angle(self, site2, unit_cell, origin):
        '''
        returns the angle between one lattice site and another lattice site,
        relative to an origin and a given crystal description
        '''
        import cryspy.util as util
        import numpy as np
        
        if isinstance(site2, lattsite) and isinstance(origin, lattsite) and \
            isinstance(unit_cell, unitcell):

            pp1 = site2.x - origin.x
            pp2 = site2.y - origin.y
            pp3 = site2.z - origin.z
            qq1 =  self.x - origin.x
            qq2 =  self.y - origin.y
            qq3 =  self.z - origin.z

            m = unit_cell.m
            nrm = 1.0 / (util.vecarraynorm([pp1, pp2, pp3]) * \
                         util.vecarraynorm([qq1, qq2, qq3]))

            return np.arccos(nrm *
                util.xtaldot(p1=pp1, p2=pp2, p3=pp3,
                        g11=m[0,0], g12=m[0,1], g13=m[0,2],
                        g21=m[1,0], g22=m[1,1], g23=m[1,2],
                        g31=m[2,0], g32=m[2,1], g33=m[2,2],
                        q1=qq1, q2=qq2, q3=qq3))

#-------------------------------------------------------------------------------

    def __sub__(self,vec2):
        if isinstance(vec2, lattsite):
            return lattsite(self.x-vec2.x, self.y-vec2.y, self.z-vec2.z)

#-------------------------------------------------------------------------------

    def __add__(self,vec2):
        if isinstance(vec2, lattvec):
            return lattsite(self.x+vec2.x,self.y+vec2.y,self.z+vec2.z)

#-------------------------------------------------------------------------------

    def __repr__(self):
    # TODO: Improve appearance and limit number of outputs
        import numpy as np
        qlist=np.array([self.x,self.y,self.z]).T

        print '\n Lattice site coordinates [x y z]'
        np.set_printoptions(precision=3,suppress=True)
        if np.shape(qlist)[0]==3:
            return ' '.join(str(x) for x in qlist)
        else:
            return '\n'.join(str(x) for x in qlist)

#-------------------------------------------------------------------------------

    def __getitem__(self,index,maxval=5):

        return lattsite(self.x[index], self.y[index], self.z[index])

#-------------------------------------------------------------------------------

class lattvec(object):
    '''
    Lattice Vector class

    REFERENCES:

    - M. De Graef, M. E. McHenry, "Structure of Materials: An Introduction to
      Crystallography, Diffraction, and Symmetry." New York, NY: Cambridge
      University Press, 2007.

    - B. D. Cullity, S. R. Stock, "Elements of X-Ray Diffraction." 3rd Ed. Upper
      Saddle River, NJ: Prentice Hall, 2001.

    - D. E. Sands, "Introduction to Crystallography." Mineaola, NY:
      Courier Dover Publications, 1993. p68, Eq. 3.3.
    '''

    def __init__(self, u=0, v=0, w=0):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type u=..., v=..., w=... etc.
        import numpy as np
        import cryspy.util as util

        # check that the shapes of all are the same
        if np.shape(u)==np.shape(v)==np.shape(w):

            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...
            self.u = util.vecarrayconvert(u)
            self.v = util.vecarrayconvert(v)
            self.w = util.vecarrayconvert(w)

            #nrm=1.0/arraynorm([u,v,w])
            #self.u_ = self.u * nrm
            #self.v_ = self.v * nrm
            #self.w_ = self.w * nrm

        else:
            print "lattvec construction error: check that the lengths of u,"\
                  " v, and w are all the same."
            return None

#-------------------------------------------------------------------------------

    def cross(self, vec2):
        # calculate the common direction between two planes in a crystal
        # NOTE: The correct cross product formula has a factor related to the
        #       volume of the unit cell. This is not normally of interest for
        #       our purposes and is neglected here. See De Graef & McHenry
        #       for details.
        import cryspy.util as util
        if isinstance(vec2, lattvec):
            u, v, w = util.vecarraycross(
                            [self.u, self.v, self.w],
                            [vec2.u, vec2.v, vec2.w])
            return lattvec(u, v, w)


#-------------------------------------------------------------------------------

    def dot(self, q2):
        '''
        Dot product of lattice vectors
        '''

        if isinstance(q2, lattvec):
            # dot product
            u = self.u * q2.u
            v = self.v * q2.v
            w = self.w * q2.w
            return lattvec(u, v, w)

#-------------------------------------------------------------------------------

    def rotate(self, q2):
        '''
        rotate lattice vector using quaternion rotation object
        '''
        from cryspy.rot import quat
        import numpy as np

        if isinstance(q2, quat):
            t2 =   q2.a * q2.b
            t3 =   q2.a * q2.c
            t4 =   q2.a * q2.d
            t5 =  -q2.b * q2.b
            t6 =   q2.b * q2.c
            t7 =   q2.b * q2.d
            t8 =  -q2.c * q2.c
            t9 =   q2.c * q2.d
            t10 = -q2.d * q2.d

            v1new = np.array([2.0 * ( (t8 + t10) * self.u +
                                      (t6 -  t4) * self.v +
                                      (t3 +  t7) * self.w ) + self.u])
            v2new = np.array([2.0 * ( (t4 +  t6) * self.u +
                                      (t5 + t10) * self.v +
                                      (t9 -  t2) * self.w ) + self.v])
            v3new = np.array([2.0 * ( (t7 -  t3) * self.u +
                                      (t2 +  t9) * self.v +
                                      (t5 +  t8) * self.w ) + self.w])

            return lattvec(u=v1new, v=v2new, w=v3new)

#-------------------------------------------------------------------------------

    def to_cartesian(self, unit_cell):
        '''
        multiply by direct structure matrix to get cartesian vector
        '''
        import cryspy.util as util
        if isinstance(unit_cell, unitcell):
            
            d = unit_cell.d
            vx = d[0, 0] * self.u + d[0, 1] * self.v + d[0, 2] * self.w
            vy = d[1, 0] * self.u + d[1, 1] * self.v + d[1, 2] * self.w
            vz = d[2, 0] * self.u + d[2, 1] * self.v + d[2, 2] * self.w
            nrm = 1.0 / util.vecarraynorm([vx, vy, vz])
            
            return [vx, vy, vz] * nrm

#-------------------------------------------------------------------------------

    @classmethod
    def from_cartesian(cls,arg,unit_cell,maxval=9):
        '''
        multiply by inverse transpose direct structure matrix
        '''
        import cryspy.util as util
        if isinstance(unit_cell, unitcell):
            x = arg[0]
            y = arg[1]
            z = arg[2]
            d = unit_cell.dinv.T
            un = d[0, 0] * x + d[0, 1] * y + d[0, 2] * z
            vn = d[1, 0] * x + d[1, 1] * y + d[1, 2] * z
            wn = d[2, 0] * x + d[2, 1] * y + d[2, 2] * z
            u, v, w, dev = util.rationalize([un, vn, wn], maxval)

            return cls(u, v, w), dev

#-------------------------------------------------------------------------------

    def length(self, unit_cell):
        '''
        multiply by metric matrix to get cartesian vector
        '''
        import cryspy.util as util
        import numpy as np
        if isinstance(unit_cell, unitcell):
            
            m = unit_cell.m
            return np.sqrt(util.xtaldot(p1=self.u, p2=self.v, p3=self.w,
                                   g11=m[0, 0], g12=m[0, 1], g13=m[0, 2],
                                   g21=m[1, 0], g22=m[1, 1], g23=m[1, 2],
                                   g31=m[2, 0], g32=m[2, 1], g33=m[2, 2],
                                   q1=self.u, q2=self.v, q3=self.w))

#-------------------------------------------------------------------------------

    def angle(self, vec2, unit_cell):
        '''
        get angle between lattice vectors
        '''
        import cryspy.util as util
        import numpy as np
        if isinstance(vec2, lattvec) and isinstance(unit_cell,unitcell):
            m = unit_cell.m

            nrm=1.0 / (self.length(unit_cell) * vec2.length(unit_cell))
            pgq = util.xtaldot(p1=self.u, p2=self.v, p3=self.w,
                            g11=m[0, 0], g12=m[0, 1], g13=m[0, 2],
                            g21=m[1, 0], g22=m[1, 1], g23=m[1, 2],
                            g31=m[2, 0], g32=m[2, 1], g33=m[2, 2],
                            q1=vec2.u, q2=vec2.v, q3=vec2.w)

            return np.arccos(nrm * pgq)

#-------------------------------------------------------------------------------

    def __sub__(self,vec2):
        if isinstance(vec2, lattvec):
            return lattvec(self.u - vec2.u, self.v - vec2.v, self.w - vec2.w)

#-------------------------------------------------------------------------------


    def __add__(self,vec2):
        if isinstance(vec2, lattvec):
            return lattvec(self.u + vec2.u, self.v + vec2.v, self.w + vec2.w)

#-------------------------------------------------------------------------------

    def __repr__(self):
    # TODO: Improve appearance and limit number of outputs
        import numpy as np
        qlist=np.array([self.u,self.v,self.w]).T

        print '\n Lattice direction vectors [u v w]'
        np.set_printoptions(precision=3,suppress=True)
        if np.shape(qlist)[0]==3:
            return ' '.join(str(x) for x in qlist)
        else:
            return '\n'.join(str(x) for x in qlist)

#-------------------------------------------------------------------------------

    def __getitem__(self,index,maxval=5):

        return lattvec(self.u[index],self.v[index],self.w[index])

#-------------------------------------------------------------------------------

class miller:
    ''' Miller indices class

    For operations on Miller plane indices

    Parameters
    ----------
    h : n x 1 list, tuple, or numpy array
        The h coordinate(s) of the Miller indices
    k : n x 1 list, tuple, or numpy array
        The k coordinate(s) of the Miller indices
    l : n x 1 list, tuple, or numpy array
        The l coordinate(s) of the Miller indices

    Attributes
    ----------
    h : n x 1 numpy array
    k : n x 1 numpy array
    l : n x 1 numpy array
    size : int
        The number of sets of Miller indices contained in the miller object.

    Notes
    -----
    Aside from the documented methods, addition and subtraction are also
    supported.

    The methods and algorithms from this class draw heavily from the De Graef
    and McHenry text [1]_, from Cullity and Stock [2]_, and from Sands [3]_.


    References
    ----------
    .. [1] M. De Graef, M. E. McHenry, "Structure of Materials: An Introduction to
       Crystallography, Diffraction, and Symmetry." New York, NY: Cambridge
       University Press, 2007.

    .. [2] B. D. Cullity, S. R. Stock, "Elements of X-Ray Diffraction." 3rd Ed. Upper
       Saddle River, NJ: Prentice Hall, 2001.

    .. [3] D. E. Sands, "Introduction to Crystallography." Mineaola, NY:
       Courier Dover Publications, 1993. p68, Eq. 3.3.
    '''

    def __init__(self, h=0, k=0, l=1):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type u=..., v=..., w=... etc.
        import cryspy.util as util
        import numpy as np

        # check that the shapes of all are the same
        if np.shape(h)==np.shape(k)==np.shape(l):

            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...
            self.h = util.vecarrayconvert(h)
            self.k = util.vecarrayconvert(k)
            self.l = util.vecarrayconvert(l)
            self.size = np.size(h)

        else:
            print "lattvec construction error: check that the lengths of u,"\
                  " v, and w are all the same."
            return None

#-------------------------------------------------------------------------------

    def dspacing(self,unit_cell):
        ''' calculate the distance between planes in a crystal

        Parameters
        ----------
        unit_cell : unitcell object

        Returns
        -------
        d : numpy array
            distances between planes in crystal

        Examples
        --------
        >>> my_dspacings = my_miller_indices.dspacing( my_unit_cell ) # TODO: Fix example
        '''
        import numpy as np
        import cryspy.util as util
        m = unit_cell.minv  # rename for space saving purposes
        return 1.0 /np.sqrt(util.xtaldot(p1 = self.h, p2 = self.k, p3 = self.l,
                                   g11 = m[0, 0], g12 = m[0, 1], g13 = m[0, 2],
                                   g21 = m[1, 0], g22 = m[1, 1], g23 = m[1, 2],
                                   g31 = m[2, 0], g32 = m[2, 1], g33 = m[2, 2],
                                   q1 = self.h, q2 = self.k, q3 = self.l))

#-------------------------------------------------------------------------------

    def to_cartesian(self,unit_cell):
        ''' convert miller indices to cartesian vectors

        Multiplies Miller indices by the inverse direct structure matrix to
        obtain cartesian vectors.

        Parameters
        ----------
        unit_cell : unitcell object

        Returns
        -------
        x : numpy array of floats
            x coordinates of cartesian vectors
        y : numpy array of floats
            y coordinates of cartesian vectors
        z : numpy array of floats
            z coordinates of cartesian vectors

        Examples
        --------
        >>> x, y, z = my_miller_object.to_cartesian( my_unit_cell ) #TODO: Fix example
        '''
        import cryspy.util as util
        if isinstance(unit_cell, unitcell):

            d  = unit_cell.dinv
            vx = d[0,0] * self.h + d[0,1] * self.k + d[0,2] * self.l
            vy = d[1,0] * self.h + d[1,1] * self.k + d[1,2] * self.l
            vz = d[2,0] * self.h + d[2,1] * self.k + d[2,2] * self.l

            nrm = util.vecarraynorm([vx, vy, vz])
            vx  = vx / nrm
            vy  = vy / nrm
            vz  = vz / nrm

            return [vx, vy, vz]

#------------------------------------------------------------------------------

    @classmethod
    def from_cartesian(cls, arg, unit_cell, maxval=9, rationalizevals=True):
        ''' convert cartesian vectors to Miller indices

        Multiply cartesian vectors by the transpose direct structure matrix to
        obtain Miller indices

        Parameters
        ----------
        arg : list of n x 1 numpy arrays
          an (x,y,z) list of cartesian coordinates where x, y, and z are n x 1
          numpy arrays
        unit_cell : unitcell object
        maxval : int, optional
            the maximum integer value of the rational miller index (default 9)
        rationalizevals : boolean
            for whether the vectors should be converted to the closest rational
            indices (default True)

        Returns
        -------
        m : miller object
        dev : numpy array of floats
            the deviations of the rational indices from the cartesian vectors.
            Returns array of zeros if rationalize=False

        Examples
        --------
        >>> import numpy as np
        >>> n = 5
        >>> x = np.random.rand(n)
        >>> y = np.random.rand(n)
        >>> z = np.random.rand(n)
        >>> my_cartesian_list = (x, y, z)
        >>> miller.from_cartesian( my_cartesian_list, my_unit_cell ) #TODO: Fix example
        '''
        import numpy as np
        import cryspy.util as util
        if isinstance(unit_cell,unitcell):
            
            x = arg[0]
            y = arg[1]
            z = arg[2]

            nrm = util.vecarraynorm([x,y,z])
            x = x/nrm
            y = y/nrm
            z = z/nrm

            d = unit_cell.dinv

            h = d[0,0]*x + d[0,1]*y + d[0,2]*z
            k = d[1,0]*x + d[1,1]*y + d[1,2]*z
            l = d[2,0]*x + d[2,1]*y + d[2,2]*z

            if rationalizevals==True:
                h, k, l, dev = util.rationalize([h, k, l], maxval)
            else:
                dev = np.zeros(np.shape(h))

            return cls(h, k, l), dev

#------------------------------------------------------------------------------

    def angle(self, plane2, unit_cell):
        ''' calculate the angle between two planes in a crystal

        Parameters
        ----------
        plane2 : miller object
            The 2nd plane for calculation
        unit_cell : unitcell object

        Returns
        -------
        theta : numpy array of floats
            The angles between the Miller planes

        Examples
        --------
        >>> plane1.angle(plane2, my_unit_cell) #TODO: Fix example
        '''
        
        import cryspy.util as util
        import numpy as np
        
        if isinstance(unit_cell, unitcell) and isinstance(plane2, miller):
            m = unit_cell.minv

            nrm=self.dspacing(unit_cell) * plane2.dspacing(unit_cell)
            ghg = util.xtaldot(p1=self.h, p2=self.k, p3=self.l,
                            g11=m[0, 0], g12=m[0, 1], g13=m[0, 2],
                            g21=m[1, 0], g22=m[1, 1], g23=m[1, 2],
                            g31=m[2, 0], g32=m[2, 1], g33=m[2, 2],
                            q1=plane2.h, q2=plane2.k, q3=plane2.l)
            return np.arccos(nrm * ghg)

#------------------------------------------------------------------------------

    def cross(self, plane2):
        ''' calculate the common direction between two planes in a crystal

        Returns the cross product between the object itself and plane2

        Parameters
        ----------
        plane2 : miller object
            The 2nd plane for calculation

        Returns
        -------
        vec : lattvec object
            Lattice vector describing common direction between the planes

        Notes
        -----
        The correct cross product formula has a factor related to the
        volume of the unit cell. This is not normally of interest for
        our purposes and is neglected here. For more information, see [1]_

        References
        ----------
        .. [1] M. De Graef, M. E. McHenry, "Structure of Materials: An Introduction to
           Crystallography, Diffraction, and Symmetry." New York, NY: Cambridge
           University Press, 2007.

        Examples
        --------
        >>> common_dir = my_first_plane.cross( my_second_plane ) #TODO: Fix example
        '''
        import cryspy.util as util
        if isinstance(plane2, miller):
            u, v, w = util.vecarraycross(
                            [self.h,   self.k,   self.l],
                            [plane2.h, plane2.k, plane2.l])
            return lattvec(u, v, w)

#------------------------------------------------------------------------------

    def rotate(self, r):
            ''' rotate miller indices

            Rotate miller indices using quaternions or rotation matrices.

            Parameters
            ----------
            r : quat or rmat object
                object containing quaternions or rotation matrices

            Returns
            -------
            m : miller object
                rotated miller indices

            Examples
            --------
            >>> m = miller(1,0,4)
            >>> s = pointgroupelements('m-3m')
            >>> ms= m * s
            '''
            from cryspy.rot import quat, rmat
            import numpy as np
            
            if isinstance(r, quat):
                t2 =   r.a * r.b
                t3 =   r.a * r.c
                t4 =   r.a * r.d
                t5 =  -r.b * r.b
                t6 =   r.b * r.c
                t7 =   r.b * r.d
                t8 =  -r.c * r.c
                t9 =   r.c * r.d
                t10 = -r.d * r.d

                v1new = np.array([2.0 * ( (t8 +t10) * self.h +
                                          (t6 - t4) * self.k +
                                          (t3 + t7) * self.l ) + self.h])
                v2new = np.array([2.0 * ( (t4 + t6) * self.h +
                                          (t5 +t10) * self.k +
                                          (t9 - t2) * self.l ) + self.k])
                v3new = np.array([2.0 * ( (t7 - t3) * self.h +
                                          (t2 + t9) * self.k +
                                          (t5 + t8) * self.l ) + self.l])

                return miller(h=v1new, k=v2new, l=v3new)

            if isinstance(r, rmat):

                v1new = self.h * r.g11 + self.k * r.g21 + self.l * r.g31
                v2new = self.h * r.g12 + self.k * r.g22 + self.l * r.g32
                v3new = self.h * r.g13 + self.k * r.g23 + self.l * r.g33

                return miller(h=v1new, k=v2new, l=v3new)


#------------------------------------------------------------------------------

    def dot(self, plane2):
        ''' dot product of miller objects

        Returns the dot product of self with miller object plane2

        Parameters
        ----------
        plane2 : miller object

        Returns
        -------
        val : numpy array of floats
            dot product values

        Examples
        --------
        >>> dotprod = my_plane1.dot(my_plane2) #TODO: fix example
        '''
        if isinstance(plane2, miller):

            # dot product
            v = self.h * plane2.h + self.k * plane2.k + self.l * plane2.l

            return v

#------------------------------------------------------------------------------

    def symmetrize(self, symmetry):
        ''' symmetrize miller object

        Returns symmetrically equivalent Miller indices

        Parameters
        ----------
        symmetry : symm object
            The symm object contains the symmetry elements of the point group

        Returns
        -------
        s : list of miller objects
            each entry in the list is symmetrically equivalent to the original
            miller object
        '''
        return [self.rotate(item) for item in symmetry.elements]

#------------------------------------------------------------------------------

    def to_fundzone(self, symmetry, unit_cell, fundzone=None):
        ''' put miller object into the fundamental zone

        Parameters
        ----------
        symmetry : symm object
        unit_cell : unitcell object
        fundzone : quat object (optional)
            Quaternion describing rotation required to put the Miller indices
            into the desired fundamental zone. Defaults to identity quaternion.

        Returns
        -------
        mf : miller object
            miller indices projected into the fundamental zone for the given
            symmetry

        Examples
        --------
        >>> import numpy as np
        >>> from cryspy.util import stereoproj
        >>> pgn = 'm-3m'
        >>> s   = symm(pgn)
        >>> pointgroupnumber = interpret_point_group_name(pgn)
        >>> maxval  = 49
        >>> numvals = 500
        >>> h  = maxval * np.random.rand(numvals) - maxval / 2.0
        >>> k  = maxval * np.random.rand(numvals) - maxval / 2.0
        >>> l  = maxval * np.random.rand(numvals) - maxval / 2.0
        >>> m  = miller(h, k, l) # create miller
        >>> uc = unitcell()
        >>> m2 = m.to_fundzone(s, uc)
        >>> pf = stereoproj()
        >>> pf.add_miller(m,  uc, uppermarkerfacecolor='r',
        ...                       lowermarkerfacecolor='w',
        ...                       uppermarkeredgecolor='r',
        ...                       lowermarkeredgecolor='r')
        >>> pf.add_miller(m2, uc, uppermarkerfacecolor='b',
        ...                       lowermarkerfacecolor='w',
        ...                       uppermarkeredgecolor='b',
        ...                       lowermarkeredgecolor='b')
        '''
        import inspect
        import cryspy.rot as rot
        import cryspy.util as util
        import numpy as np

        if fundzone==None:
            fundzone = rot.quat()

        ms = self.symmetrize(symmetry)

        # Consolidate symmetrized list into a single array of cartesian vectors
        x = np.zeros([self.size, np.size(ms)])
        y = np.zeros([self.size, np.size(ms)])
        z = np.zeros([self.size, np.size(ms)])
        k = 0
        for item in ms:
            tmp = item.to_cartesian(unit_cell)
            x[0:self.size, k] = tmp[0]
            y[0:self.size, k] = tmp[1]
            z[0:self.size, k] = tmp[2]
            k += 1

        # Normalize cartesian vectors
        nrm = 1.0 / util.vecarraynorm([x, y, z])
        x = x * nrm
        y = y * nrm
        z = z * nrm

        # Convert to spherical coordinates
        theta, r, rho = util.polar(x, y, z)

        # get limits of the fundamental zone of the stereographic projection
        mintheta, maxtheta, minrho, maxrho = fundzonePF(
                                                   symmetry.point_group_number)

        # handle theta cases on edges of fundamental zone
        cutoff = np.sqrt(np.spacing(1))
        loc = np.absolute(theta - mintheta) < cutoff
        theta[loc] = mintheta
        loc = np.absolute(theta - maxtheta) < cutoff
        theta[loc] = maxtheta

        if inspect.isfunction(maxrho):
            mxr = maxrho(theta)
            loc = np.absolute(rho - minrho) < cutoff
            rho[loc] = minrho
            loc = np.absolute(rho - mxr)    < cutoff
            rho[loc] = mxr[loc]
            crt = np.array(theta <= maxtheta) * np.array(rho <= mxr) * \
                  np.array(theta >= mintheta) * np.array(rho >= minrho)
        else:
            crt = np.array(theta <= maxtheta) * np.array(rho <= maxrho) * \
                  np.array(theta >= mintheta) * np.array(rho >= minrho)
        loc = np.argmax(crt, axis=1)

        # extract the symmetrically-equivalent miller indices in the fundamental zone
        h = np.zeros(self.size)
        k = np.zeros(self.size)
        l = np.zeros(self.size)
        
        for i in range(0, np.size(loc)):
            h[i] = np.squeeze(ms[loc[i]].h)[i]
            k[i] = np.squeeze(ms[loc[i]].k)[i]
            l[i] = np.squeeze(ms[loc[i]].l)[i]

        # create a new miller object and rotate it into the desired zone
        return miller(h,k,l).rotate(fundzone)

#------------------------------------------------------------------------------

    def __sub__(self, vec2):
        if isinstance(vec2, miller):
            return miller(self.h - vec2.h, self.k - vec2.k, self.l - vec2.l)

#------------------------------------------------------------------------------

    def __add__(self, vec2):
        if isinstance(vec2, miller):
            return miller(self.h + vec2.h, self.k + vec2.k, self.l + vec2.l)

#-------------------------------------------------------------------------------

    def __getitem__(self,index):
            return miller(self.h[index], self.k[index], self.l[index])

#-------------------------------------------------------------------------------

    def __repr__(self):
    # TODO: Improve appearance and limit number of outputs
        import numpy as np
        qlist=np.array(np.squeeze([self.h, self.k, self.l])).T

        print '\n Miller indices (h k l)'
        np.set_printoptions(precision=3, suppress=True)
        if np.shape(qlist)[0]==3:
            return ' '.join(str(x) for x in qlist)
        else:
            return '\n'.join(str(x) for x in qlist)

#-------------------------------------------------------------------------------

class unitcell(object):
    """
    Unit Cell class

    Initializes a unit cell structure. Calculates the metric matrix and volume.

    References
    ----------
    .. [1] M. De Graef, M. E. McHenry, "Structure of Materials: An Introduction to
       Crystallography, Diffraction, and Symmetry." New York, NY: Cambridge
       University Press, 2007.

    .. [2] B. D. Cullity, S. R. Stock, "Elements of X-Ray Diffraction." 3rd Ed. Upper
       Saddle River, NJ: Prentice Hall, 2001.

    .. [3] D. E. Sands, "Introduction to Crystallography." Mineaola, NY:
       Courier Dover Publications, 1993. p68, Eq. 3.3.
    """

    def __init__(self, a=1.0, b=1.0, c=1.0,
                       alpha=90.0, beta=90.0, gamma=90.0):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type g11=..., g12=... etc.
        import numpy as np
        import numpy.linalg as npla
        import cryspy.util as util

        # check that the shapes of all are the same
        if np.shape( a)==np.shape( b)==np.shape( c)==\
           np.shape(alpha)==np.shape(beta)==np.shape(gamma):

            # FIXME: Can the following be done with util.vecarrayconvert???

            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...
            self.a = np.float64( np.squeeze( np.array( list( (a,) ))))
            self.b = np.float64( np.squeeze( np.array( list( (b,) ))))
            self.c = np.float64( np.squeeze( np.array( list( (c,) ))))
            self.alpha = util.radians(np.float64( np.squeeze( np.array( list( (alpha,) )))))
            self.beta  = util.radians(np.float64( np.squeeze( np.array( list( (beta,) )))))
            self.gamma = util.radians(np.float64( np.squeeze( np.array( list( (gamma,) )))))

            # calculate the metric matrix and its inverse
            ca = np.cos(self.alpha)
            cb = np.cos(self.beta)
            cg = np.cos(self.gamma)
            self.m = np.array([[self.a * self.a,
                              self.a * self.b * cg,
                              self.a * self.c * cb],
                             [self.b * self.a * cg,
                              self.b * self.b,
                              self.b * self.c * ca],
                             [self.c * self.a * cb,
                              self.c * self.b * ca,
                              self.c**2.0]])
            self.minv = npla.inv(self.m)

            # calculate the unit cell volume
            self.vol=self.a * self.b * self.c * np.sqrt(1.0 -
                np.cos(self.alpha)**2.0 - np.cos(self.beta)**2.0 - np.cos(self.gamma)**2.0
                + 2.0 * np.cos(self.alpha) * np.cos(self.beta) * np.cos(self.gamma))

            # calculate the direct structure matrix
            ff=cb*cg-ca
            sg=np.sin(self.gamma)
            sgi=1.0/sg
            self.d=np.array([[self.a, self.b*cg,  self.c*cb                  ],
                          [   0.0, self.b*sg, -self.c*ff*sgi              ],
                          [   0.0,       0.0,  self.vol/(self.a*self.b*sg)]])
            self.dinv=npla.inv(self.d).T

    def __repr__(self):
    # TODO: Improve appearance of output
        import numpy as np
        qlist=np.array([self.a,self.b,self.c,self.alpha,self.beta,self.gamma]).T

        print '\n Unit cell a, b, c, alpha, beta, gamma'
        np.set_printoptions(precision=3,suppress=True)
        if np.shape(qlist)[0]==3:
            return ' '.join(str(x) for x in qlist)
        else:
            return '\n'.join(str(x) for x in qlist)

#------------------------------------------------------------------------------

class symm(object):
    """symmetry object
    
    Wraps up the symmetry elements for a given point group into a single
    object. Pure rotations are described in quaternion form while the complete
    set of symmetry operations (i.e., including improper rotations) is
    given as rotation matrices.
    
    Parameters
    ----------
    pointgroup : str
        point group of the crystal
    notation : {'international', 'schoenflies', 'geo', 'tsl', 'numeric'}
        Default: 'international'
    
    Notes
    -----
    - No .size method is provided for this class because the number of symmetry
      elements is different from the number of rotation operations. Both of
      these parameters are stored within the symm object and have their own
      .size methods.    
    """
    
    def __init__(self, pointgroup='1', notation='international'):
        import cryspy.rot as rot

        # Make sure the arguments are strings
        self.notation = str(notation)
        if notation=='numeric': # convert to international
            self.point_group_name = interpret_point_group_number(pointgroup)
            self.notation = 'international'
        else:
            self.point_group_name = str(pointgroup)
        
        # Use string arguments to construct symm object attributes
        self.point_group_number = interpret_point_group_name(
                                             self.point_group_name,
                                             self.notation )
        self.elements = pointgroupelements(
                                             self.point_group_name,
                                             self.notation )
        self.rotations = rot.quat.from_rmat( rotationelements(
                                             self.point_group_name,
                                             self.notation ))

    def __repr__(self):
        return '\n Symmetry object of point group: ' \
               + str(self.point_group_name) + '\n'

    def laueclass(self):
        """ returns the laue group of the point group """
        lcn = laueclass(self.point_group_name,  self.notation)
        nom = interpret_point_group_number(lcn, self.notation)
        return symm(nom, self.notation)

#------------------------------------------------------------------------------

class orientation(object):
    """
    a crystal orientation is defined in cryspy as a rotation relative to a
    reference frame, including symmetry
    """
    def __init__(self, quaternions=None,
                 pointgroupnumbers=None, reference=None):
        import cryspy.rot as rot
        import cryspy.xtal as xtal
        import numpy as np
        import warnings

        if reference==None:
            reference = rot.quat()

        if quaternions==None:
            quaternions = rot.quat()

        if pointgroupnumbers==None:
            pointgroupnumbers = 1

        if isinstance(quaternions, rot.quat):
            self.rotations = quaternions
        else:
            self.rotations = rot.quat()


        loc = np.where(np.logical_or(pointgroupnumbers > 32, pointgroupnumbers < 1))
        pointgroupnumbers[loc] = 1
        warnings.warn('Invalid point group numbers at:\n{0}'.format(loc))
        self.symmetry = pointgroupnumbers

        if isinstance(reference, rot.quat):
            self.reference = reference
        
        self.size = self.rotations.size

        # Add two parameters to the orientation object. One is a list of
        # quaternion objects corresponding to each crystal symmetry identified.
        # The other is a list of the orientation array locations corresponding to
        # each symmetry.
        u = np.unique(self.symmetry)
        cs = []
        cslocs = []
        for item in u:
            cs_set = xtal.symm(item, 'numeric')
            cs.append(cs_set)
            cslocs = np.where(self.symmetry==item)
        self.cs = cs
        self.cslocs = cslocs

    def __getitem__(self,index):
            return orientation(quaternions = self.rotations[index],
                               pointgroupnumbers = self.symmetry[index],
                               reference = self.reference())

    def __repr__(self):
        
        import numpy as np

        ar= self.reference.a[0]
        br= self.reference.b[0]
        cr= self.reference.c[0]
        dr= self.reference.d[0]

        repstr = '{0:^45s}\n'.format('orientation object')
        repstr += '{0:-^45s}\n'.format('----')
        repstr += ' {0:^7s}{1:^7s}{2:^7s}{3:^7s}   # {4:<s}\n'.format(        \
                                               'a', 'b','c', 'd','point group')
        
        fmtstr = '<{0: >7.3f}{1: >7.3f}{2: >7.3f}{3: >7.3f} >' + \
                 ' # {4: >6.0f}\n'

        if np.shape(self.rotations.a)[0] > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.rotations.a[i], 
                                        self.rotations.b[i], 
                                        self.rotations.c[i], 
                                        self.rotations.d[i], 
                                        self.symmetry[i])
            for i in np.arange(0, 3):
                repstr += '  {0:^7s}{0:^7s}{0:^7s}{0:^7s}  \n'.format('.')
            n = np.shape(self.reference.a)[0]-1
            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.rotations.a[i], 
                                        self.rotations.b[i], 
                                        self.rotations.c[i], 
                                        self.rotations.d[i], 
                                        self.symmetry[i])    

        else:

            
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.rotations.a[i], 
                                        self.rotations.b[i], 
                                        self.rotations.c[i], 
                                        self.rotations.d[i], 
                                        self.symmetry[i])
                                        
        repstr += '{0:-^45s}\n'.format('----')                    
        namestr = 'reference q = '
        namestr += '<{0: >7.3f}{1: >7.3f}{2: >7.3f}{3: >7.3f} >'.format(ar, br,
                                                                        cr, dr)
        repstr += '{0:<s}\n'.format(namestr)                          
        repstr += '\n'
        return repstr

# -----------------------------------------------------------------------------

def ipfgrid(pointgroupnumber, resolution=0.5, uc=None, trans='ea',
            mintheta=0, maxtheta=None, maxrho=None, fundzone=None):
    ''' Generate a grid in the fundamental zone for the Laue group

    Parameters
    ----------
    pointgroup : int, range 1:32
        the point group number

    uc : unitcell object (optional)
        Defaults to the cubic case.

    resolution : float, degrees (optional)
        the approximate angular step size along the mintheta direction (the
        actual step size will varys in order to fill the complete fundamental
        zone). Default = 0.5 degrees.

    trans : {'ea', 'stereo'} (optional)
        the transformation to be applied (either 'ea' for equal area or
        'stereo' for stereographic). Default='stereo'

    mintheta : float, radians (optional)
        the minimum theta value in polar coordinates for the fundamental zone.

    maxtheta : float, radians (optional)
        the maximum theta value in polar coordinates for the fundamental zone.

    maxrho : float, radians (optional)
        the maximum value of rho in polar coordinates for the fundamental zone.

    fundzone : quat object (optional)
        a quaternion describing the specific zone to be used

    Returns
    -------
    xp    : numpy array of floats
        the x coordinates of the grid points in the given projection
    yp    : numpy array of floats
        the y coordinates of the grid points in the given projection
    hem   : numpy array of char
        the hemisphere of the given grid points {'N','S'}
    edges : numpy array of int
        indices of the outer edge points in the xp, yp, and hem arrays

    Notes
    -----
    - The code currently automatically uses the fundamental zone of the Laue
      group, as is common for EBSD. Thus, inverse pole figures may not be
      strictly correct for some symmetries.
    - The code also does not automatically adjust for various settings for
      monoclinic, etc., even though a good guess might be made using the
      unit cell if it is specified. These would be very nice additions for
      future versions of the code.
    - When maxtheta and maxrho are given, then the Laue group fundamental
      zone is overridden. Thus, if the user knows exactly what the necessary
      shape of their fundamental zone is, then they can use what is correct
      for their phase. This override does not currenlty work for cases when
      rho is a function of theta. This should not generally be a problem
      unless there exist some cases where the cubic or rhombohedral
      fundamental zones need to be overridden.
    - The algorithm used is a simplified translation of the one used in the
      mtex toolbox for matlab [1]_.
    - In EBSD, it is probably possible to determine the point group for
      some phases rather than the Laue group. The gridding below is INCORRECT
      for the point groups in general, because it won't show differences when
      there is no center of symmetry.
    - According to reference [2]_, there are at least 10 non-standard
      settings for triclinic, 30 for monoclinic, and 165 for orthorhombic.
      Coding a general solution would therefore likely be time consuming...

    References
    ----------
    .. [1] R. Hielscher, F. Bachmann, H. Schaeben. "MTEX: A MATLAB Toolbox for
       Quantitative Texture Analysis. Version 3.3.1.
    .. [2] M. Darby Dyar, M. E. Gunter. Mineralogy and Optical Mineralogy, Ch. 12.
       Mineralogical Society of America 1997-2013.

    Examples
    --------
    >>> xp,yp,hem = ipfgrid(17)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.plot(xp, yp,'.')
    >>> plt.axis('equal')
    '''
    import inspect
    import cryspy.rot as rot
    import cryspy.util as util
    import numpy as np
    
    mq = mintheta; del mintheta # rename so we can reuse 'mintheta'

    if uc==None:
        uc = unitcell() # initialize unitcell object if necessary

    if fundzone==None:
        fundzone = rot.quat()

    override_flag = False
    if maxtheta != None or maxrho != None:
        override_flag = True

    lc = laueclass(pointgroupnumber)
    resolution = resolution * np.pi / 180.0

    theta = np.array([])
    rho   = np.array([])
    edges = np.array([])

    mintheta, maxtheta, minrho, maxr = fundzonePF(lc)

    if inspect.isfunction(maxr):

        res = (maxtheta - mintheta) / \
              np.ceil( (maxtheta - mintheta) / resolution)
        q = np.arange(mintheta + mq, maxtheta + mq + res, res + np.sqrt(np.spacing(1)))

        for val in q:

            maxrho = maxr(val)

            res = maxrho / np.ceil(maxrho / resolution)
            p = np.arange(minrho, maxrho + res, res + np.sqrt(np.spacing(1)))
            theta = np.append(theta, np.tile(val, np.size(p)))
            rho = np.append(rho, p)
            edges = np.append(edges, np.size(theta))

    else:
        res = (maxtheta - mintheta) / \
              np.ceil( (maxtheta - mintheta) / resolution)
        q = np.arange(mintheta + mq, maxtheta + mq + res, res + np.sqrt(np.spacing(1)))
        res = maxr / np.ceil(maxr / resolution)
        p = np.arange(minrho, maxr + res, res + np.sqrt(np.spacing(1)))
        qn = np.size(q)
        pn = np.size(p)

        theta = np.reshape(np.tile(q, [pn, 1]).T, -1)
        rho   = np.reshape(np.tile(p, [qn, 1]),   -1)
        edges = np.tile(qn,np.size(theta))

    if override_flag == True: # custom
        res = (maxtheta - mintheta) / \
              np.ceil( (maxtheta - mintheta) / resolution)
        q = np.arange(mintheta, maxtheta + res, res + np.sqrt(np.spacing(1)))
        res = maxrho / np.ceil(maxrho / resolution)
        p = np.arange(minrho, maxrho + res, res + np.sqrt(np.spacing(1)))
        qn = np.size(q)
        pn = np.size(p)

        theta = np.reshape(np.tile(q, [pn, 1]).T, -1)
        rho   = np.reshape(np.tile(p, [qn, 1]),   -1)
        edges = np.tile(qn,np.size(theta))

    x = np.sin(rho) * np.cos(theta)
    y = np.sin(rho) * np.sin(theta)
    z = np.cos(rho)

    if trans=='ea':
        xp, yp, hem = util.eatransstandard([x, y, z])
    elif trans=='stereo':
        xp, yp, hem = util.stereotransstandard([x, y, z])
    else:
        # Assume the user wants equal area if they don't want the default
        print 'Requested projection transformation not recognized. ' \
              'Proceeding with equal area.\n'
        xp, yp, hem = util.eatransstandard([x, y, z])

    return xp, yp, hem, edges

# -----------------------------------------------------------------------------

def fundzonePF(pointgroupnumber):
    '''
    Returns the limits of the standard fundamental zone of the Laue class of
    the provided point group, in polar coordinates.

    It would be useful to extend this in the future to cover different settings
    as well as the correct fundamental zones for each symmetry, rather than
    return the Laue group. Returning the Laue group is currently common in
    EBSD, but may not be necessary.

    Refer to warnings and documentation in ipfgrid for more info.

    See documentation for util/polar2d for info about the convention used for
    polar coordinates.
    '''

    lc = laueclass(pointgroupnumber) # it would be best if we can set up all of the
                               # conditions such that we can remove this step!

    #--------------------------------------------------------------------------
    # Set up definitions for cases where the maxrho is a function of theta
    def maxrho29(q):
        # maximum value of rho for point group 29 ('m-3')
        import numpy as np

        if np.size(q)>1:
            mp = np.zeros(np.shape(q))

            loc = np.array(q < np.pi/4.0)
            mp[loc] = np.pi - \
                      np.arctan2(np.cos(np.pi/4.0),
                                 np.sin(np.pi/4.0) * np.cos(q[loc] + np.pi))

            loc = np.array(q >= np.pi/4.0) * np.array(q <= 3.0/4.0*np.pi)
            mp[loc] = np.pi - \
                      np.arctan2(np.cos(np.pi/4.0),
                                np.sin(np.pi/4.0) * np.cos(q[loc] + np.pi/2.0))


            loc = np.array(q > np.pi*3.0/4.0)
            mp[loc] = np.pi - \
                      np.arctan2(np.cos(np.pi/4.0),
                                 np.sin(np.pi/4.0) * np.cos(q[loc]))
        else:
            if   q < np.pi/4.0:
                mp = np.pi - \
                     np.arctan2(np.cos(np.pi/4.0),
                                np.sin(np.pi/4.0) * np.cos(q + np.pi))
            elif (q >= np.pi/4.0) or (q <= 3.0/4.0*np.pi):
                mp = np.pi - \
                     np.arctan2(np.cos(np.pi/4.0),
                                np.sin(np.pi/4.0) * np.cos(q + np.pi/2.0))
            elif q > np.pi*3.0/4.0:
                mp = np.pi - \
                     np.arctan2(np.cos(np.pi/4.0),
                                np.sin(np.pi/4.0) * np.cos(q))

        return mp

    def maxrho32(theta):
        # maximum value of rho for point group 32 ('m-3m')
        import numpy as np
        return np.pi - np.arctan2(np.cos(np.pi / 4.0),\
                      np.sin(np.pi / 4.0) * \
                      np.cos(np.mod(theta, 2.0 * maxtheta) + np.pi))

    #--------------------------------------------------------------------------

    # Proceed setting up maximum values for individual Laue group cases
    import numpy as np
    mintheta = 0.0 #+ np.spacing(1)
    minrho = 0.0

    if lc == 32: # m-3m

        maxtheta = 45.0*np.pi/180.0
        maxrho = maxrho32

    if lc == 29:

        maxtheta = np.pi/2.0
        maxrho = maxrho29

    if lc == 27: # 6/mmm

        maxtheta = np.pi/6.0
        maxrho   = np.pi/2.0

    if lc == 23 or lc == 20: # m-3, 6/m, and -3m, respectively

        maxtheta = np.pi/3.0
        maxrho   = np.pi/2.0

    if lc == 17: # -3

        maxtheta = 2.0*np.pi/3.0
        maxrho   = np.pi/2.0

    if lc == 15: # 4/mmm

        maxtheta = np.pi/4.0
        maxrho   = np.pi/2.0

    if lc == 11 or lc == 8: # 4/m or mmm, respectively

        maxtheta = np.pi/2.0
        maxrho   = np.pi/2.0

    if lc == 5: # 2/m

        maxtheta = np.pi
        maxrho   = np.pi/2.0

    if lc == 2: #-1

        maxtheta = 2.0*np.pi
        maxrho   = np.pi/2.0

    return mintheta, maxtheta, minrho, maxrho

# -----------------------------------------------------------------------------
def iscentrosymmetric(pointgroupnumber):
    ''' center of symmetry check
    
    Returns true if the point group has a center of symmetry and false if it
    does not.
    
    Parameters
    ----------
    pointgroupnumber : int numpy array
        point group number
    
    Returns
    -------
    b : Boolean
        True/false list
    '''
    import numpy as np
    b = np.in1d(pointgroupnumber, [2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32])
    return b[0]