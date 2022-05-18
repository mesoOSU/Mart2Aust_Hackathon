# -*- coding: utf-8 -*-
'''
ovlib.rot: Rotation descriptions and conversions
===============================================================================
'''

class angax(object):
    """ 
    Angle/axis rotation representation.
    
    ...
    
    Attributes
    ----------
    th : list, tuple, or numpy array of floats
        the angle in radians
    x  : list, tuple, or numpy array of floats
        the axis x coordinate
    y  : list, tuple, or numpy array of floats
        the axis y coordinate
    z  : list, tuple, or numpy array of floats
        the axis z coordinate
    
    Notes
    -----
    Axes are normalized during initialization
    """
    
    def __init__(self,th=1,x=0,y=0,z=0):
        import numpy as np
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type a=..., b=... etc.
        
        # check that the shapes of all are the same
        if np.shape(th)==np.shape(x)==np.shape(y)==np.shape(z):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            th = np.asarray(th)
            x  = np.asarray(x)
            y  = np.asarray(y)
            z  = np.asarray(z)

            nrm=np.array([x, y, z])
            nrm=1.0 / np.sum(nrm**2.0, axis=0)

            self.th = th
            self.x = x * nrm
            self.y = y * nrm
            self.z = z * nrm
            self.size = np.size(self.x)
        
        else:
            print "angax construction error: check that the lengths of th, x,"\
                  " y, and z are all the same."
            return None

#-------------------------------------------------------------------------------

    def numel(self):
        """ returns angax size
        
        Gives the number of elements in the angle/axis rotation object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        n : int
            the number of elements in the rotation matrix object
        
        Examples
        --------
        >>> import numpy as np
        >>> import cryspy.rot as rot
        >>> my_angax = rot.angax(th=[np.pi/7.0, np.pi], 
        ...                     x=[1, 2], y=[1, 1], z=[1, 5])
        >>> numel = my_angax.numel()
        """
        return self.x.shape[0]

#-------------------------------------------------------------------------------

    def to_array(self):
        ''' converts object to array
        
        Converts angax object to an n x 4 numpy array where the first column
        corresponds to the angle and the final three columns correspond to the
        axis.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        a : angax object
            object containing the angle/axis representation of the rotation

        Examples
        --------
        >>> import cryspy.rot as rot
        >>> my_angax = rot.angax(th=0.785, x=1.0, y=1.0, z=0.0)
        >>> my_array = my_angax.to_array()
        
        See Also
        --------
        np.array : numpy array
        '''
        import numpy as np
        return np.vstack([self.x, self.y, self.z]).T
        
#-------------------------------------------------------------------------------
    
    @classmethod
    def from_array(cls, arg):
        ''' create angax object from numpy array
        
        Converts an n x 4 numpy array where the first column corresponds to
        angles (in radians) and the latter three columns correspond to the
        axes into an angax object.

        Parameters
        ----------
        arg : n x 4 numpy array
            array to be converted into angax object
        
        Returns
        -------
        a : angax object
            object containing the angle/axis representation of the rotation
        
        Examples
        --------
        >>> import numpy as np
        >>> import cryspy.rot as rot
        >>> my_array = [[np.pi, 1.0, 0.0, 0.0], [np.pi/4.0, 0.0, 1.0, 3.0]]
        >>> my_angax = rot.angax.from_array( my_array )  
        
        See Also
        --------
        np.array : numpy array
        '''
        import numpy as np
        s = arg.shape()
        if isinstance(arg, np.array) and np.any(np.asarray(s)==4):
            if s[1] == 4:
                return cls(arg[:,0], arg[:,1], arg[:,2], arg[:,3])
            if s[0] == 4:
                arg = arg.T
                return cls(arg[:,0], arg[:,1], arg[:,2], arg[:,3])
        else:
            print 'input is not an n x 4 numpy array'

#-------------------------------------------------------------------------------
            
    @classmethod
    def from_bunge(cls, arg):
        ''' convert bunge to angax
        
        Converts Bunge Euler angle object to angle/axis representation.
        
        Parameters
        ----------
        arg : Bunge object containing Euler angles
        
        Returns
        -------
        a : angax object
            object containing the angle/axis representation of the rotation
        
        Examples
        --------
        >>> import cryspy.rot as rot
        >>> import numpy as np
        >>> my_euler = rot.bunge(phi1=[np.pi/9.0, np.pi/3.0],
        ...                      PHI =[np.pi/2.0, np.pi/8.0],
        ...                      phi2=[np.pi/4.0, np.pi/6.0])
        >>> my_angax = rot.angax.from_bunge( my_euler )
        
        See Also
        --------
        bunge : Bunge Euler angle rotation object
        '''
        return angax.from_quat(quat.from_bunge(arg))

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_quat(cls, arg):
        ''' convert quat to angax
        
        Converts quaternions to angle/axis representation.
        
        Parameters
        ----------
        arg : quaternion object
        
        Returns
        -------
        a : angax object
            object containing the angle/axis representation of the rotation

        Examples
        --------
        >>> import numpy as np
        >>> import cryspy.rot as rot
        >>> my_quat  = rot.bunge(a=[np.pi, np.pi/6.0],
        ...                      b=[1.0, 0.0],
        ...                      c=[0.0, 1.0],
        ...                      d=[0.0, 0.0])
        >>> my_angax = rot.angax.from_quat( my_quat )
        
        See Also
        --------
        rodri : Rodrigues vector object
        '''
        import numpy as np
        
        aa=arg.a
        aa[aa>1.0]=1.0
        
        an = 2.0 * np.arccos(aa)
        
        # angle/axis is not well-defined when angle approaches zero        
        loc = an >= np.sqrt(np.spacing(1))  
        sqi = np.zeros(np.size(an))               
        sqi[loc]= 1.0 / np.sin(0.5 * an[loc])
        
        # assign axes
        xx = arg.b * sqi
        yy = arg.c * sqi
        zz = arg.d * sqi
        
        # do not allow rotations greater than pi
        loc = an > np.pi
        an[loc] = 2.0 * np.pi - an[loc]
        xx[loc] = -xx[loc]
        yy[loc] = -yy[loc]
        zz[loc] = -zz[loc]
        
        # when the angle is near zero, force the axis to be 1 0 0
        fix = an < np.sqrt(np.spacing(1))
        an[fix] = 0.0
        xx[fix] = 1.0
        yy[fix] = 0.0
        zz[fix] = 0.0
        
        # when the angle is very close to pi, make it pi
        fix = np.absolute(np.pi - an) < np.sqrt(np.spacing(1))
        an[fix] = np.pi
        
        return cls(th=an,x=xx,y=yy,z=zz)

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_rodri(cls, arg):
        ''' convert rodri to angax
        
        Converts Rodrigues vectors to angle/axis representation
        
        Parameters
        ----------
        arg : rodri object
            object containing Rodrigues vectors, see :rodri:
        
        Returns
        -------
        a : angax object
            object containing the angle/axis representation of the rotation

        Examples
        --------
        >>> import cryspy.rot as rot
        >>> my_rodri = rot.rodri()
        >>> my_angax = rot.angax.from_rodri( my_rodri ) 
        
        See Also
        --------
        rodri : Rodrigues vector object
        '''
        import numpy as np
        nrm=np.array([arg.r0, arg.r1, arg.r2, arg.r3])
        nrm=1.0 / np.sqrt(np.sum(nrm**2.0, axis=0))
        
        an=2.0 * np.arctan(nrm)   
        xx=arg.r1 * nrm
        yy=arg.r2 * nrm
        zz=arg.r3 * nrm
        
        return cls(th=an, x=xx, y=yy, z=zz)

#-------------------------------------------------------------------------------
        
    @classmethod
    def from_rmat(cls, arg):
        '''
        converts rotation matrices to angle/axis representation
        
        Parameters
        ----------
        arg : rmat object
            rotation matrix
        
        Returns
        -------
        angax : angax object
            angle/axis rotation representation                    
        
        See Also
        --------
        rmat
        
        Notes
        -----
        * assumes active rotation matrix        
        
        * adapted from Peter Kovesi's matlab code, which itself follows the
          implementation suggested by Hartley & Zissermann. 
        
          Original header from Kovesi's code follows::

            MATRIX2ANGLEAXIS - Homogeneous matrix to angle-axis description
            
            Usage: t = matrix2angleaxis(T)
            
            Argument:   T - 4x4 Homogeneous transformation matrix
            Returns:    t - 3-vector giving rotation axis with magnitude equal
                            to the rotation angle in radians.
            
            See also: ANGLEAXIS2MATRIX, ANGLEAXIS2MATRIX2, ANGLEAXISROTATE,
                      NEWANGLEAXIS, NORMALISEANGLEAXIS
            
            Copyright (c) 2008 Peter Kovesi
            School of Computer Science & Software Engineering
            The University of Western Australia
            pk at csse uwa edu au
            http://www.csse.uwa.edu.au/
            
            Permission is hereby granted, free of charge, to any person
            obtaining a copy of this software and associated documentation
            files (the "Software"), to deal in the Software without 
            restriction, subject to the following conditions:
            
            The above copyright notice and this permission notice shall be 
            included in all copies or substantial portions of the Software.
            
            The Software is provided "as is", without warranty of any kind.
        '''
        # Following the implementation suggested by Hartley and Zisserman:
        # Find rotation axis as the eigenvector having unit eigenvalue
        # -- solve (R-I)v = 0
        import numpy as np
        import numpy.linalg as npla
        n = np.size(arg.g11)
        v = np.complex128(np.zeros([n,9]))
        d = np.zeros([n,3])        
        
        for i in range(0, n):
            dtmp,vtmp = npla.eig(np.array(
                            [[arg.g11[i]-1.0, arg.g12[i],     arg.g13[i]],
                            [arg.g21[i],      arg.g22[i]-1.0, arg.g23[i]],
                            [arg.g31[i],      arg.g32[i],     arg.g33[i]-1.0]]
                            ))
                                   
            v[i] = np.array([vtmp[0,0], vtmp[0,1], vtmp[0,2],
                        vtmp[1,0], vtmp[1,1], vtmp[1,2],
                        vtmp[2,0], vtmp[2,1], vtmp[2,2]])
            
            d[i] = np.absolute(np.array([dtmp[0], dtmp[1], dtmp[2]]))
        
        
        # find the index of the smallest eigenvalue
        ind=np.argsort(d.T,axis=0).T
        
        # extract the appropriate eigenvectors
        xx=np.zeros(n)
        yy=np.zeros(n)
        zz=np.zeros(n)
        chk=np.complex128(np.zeros(n))
        for i in range(0, n):
            loc=ind[i,0]
            xx[i]=v[i,loc]
            yy[i]=v[i,loc+3]
            zz[i]=v[i,loc+6]
            chk[i]=d[i,loc] # record the min eigenvalue
        
        # DEBUG
        if any(chk > 0.001):
            print 'At least one rotation matrix is dubious.\n'
            print 'Location(s):\n'
            print np.where(chk > 0.001)
        
        chk=(xx**2.0 + yy**2.0 + zz**2.0)**0.5 - 1.0
        if any(chk > 0.0001):
            print 'At least one rotation matrix results in a non-unit axis\n'
            print 'Location(s)\n'
            print np.where(chk > 0.0001)
        
        # determine rotation angle
        tcq = arg.g11 + arg.g22 + arg.g33 - 1.0
        tsq = xx * (arg.g32 - arg.g23) + yy * (arg.g13 - arg.g31) + zz * (arg.g21 - arg.g12)
        an  = np.arctan2(tsq, tcq)
        
        # fix negative thetas
        rev = an < 0.0
        an[rev] = -an[rev]
        xx[rev] = -xx[rev]
        yy[rev] = -yy[rev]
        zz[rev] = -zz[rev]
        
        # put in the range of 0 to pi
        loc=np.absolute(an)>np.pi
        an[loc] = 2.0 * np.pi - an[loc]
        xx[loc] = -xx[loc]
        yy[loc] = -yy[loc]
        zz[loc] = -zz[loc]
        
        return angax(th=an, x=xx, y=yy, z=zz)            

#-------------------------------------------------------------------------------        

    def __repr__(self):
        import numpy as np
        repstr = '{0: ^34s}\n'.format('angle/axis pairs') + \
                 '{0:-^34s}\n'.format('-') + \
                 '{0: >8s}     {1:^6s} {2:^6s} {3:^6s} \n'.format('theta', 'x',
                                                              'y', 'z')
        fmtstr = '{0: >8.3f}\xb0 @ [{1: >6.3f} {2: >6.3f} {3: >6.3f}]\n'
        
        n = self.size - 1

        if self.size > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.th[i] * 180.0/np.pi, 
                                        self.x[i], 
                                        self.y[i], 
                                        self.z[i])
            for i in np.arange(0, 3):
                repstr += ' {0: ^7s}     {0: ^6s} {0: ^6s} {0: ^6s} \n'.format(
                                         '.')
            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.th[i] * 180.0/np.pi, 
                                        self.x[i], 
                                        self.y[i], 
                                        self.z[i])

        else:
  
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.th[i] * 180.0/np.pi, 
                                        self.x[i], 
                                        self.y[i], 
                                        self.z[i])
                                        
        repstr += '\n'
        return repstr

#-------------------------------------------------------------------------------

    def __getitem__(self,index):
        
        an = self.th[index]
        xx = self.x[index]
        yy = self.y[index]
        zz = self.z[index]
        
        return angax(th=an, x=xx, y=yy, z=zz)

#-------------------------------------------------------------------------------

class bunge(object):
    '''
    Bunge Euler angles rotation representation.
    
    ...
    
    Attributes
    ----------
    phi1 : list, tuple, or numpy array of floats
        the angle phi1 describing the first rotation (about the x reference
        axis), in radians
    PHI  : list, tuple, or numpy array of floats
        the angle PHI describing the second rotation (about the z reference
        axis), in radians
    phi2  : list, tuple, or numpy array of floats
        the angle phi2, describing the third rotation (about the x reference
        axis), in radians
    '''

    def __init__(self, phi1=0,PHI=0,phi2=0):
        import numpy as np
        import cryspy.util as util
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type phi1=..., PHI=... etc.
        
        # check that the shapes of all are the same
        if np.shape(phi1)==np.shape(PHI)==np.shape(phi2):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            self.phi1 = util.vecarrayconvert(phi1)
            self.PHI  = util.vecarrayconvert(PHI)
            self.phi2 = util.vecarrayconvert(phi2)
            self.size = np.size(self.phi1)
        
        else:
            print "bunge construction error: check that the lengths of phi1, "\
                  "PHI, and phi2 are all the same."
            return None
            
#-------------------------------------------------------------------------------

    def numel(self):
        return self.PHI.shape[0]

#-------------------------------------------------------------------------------

    def to_array(self):
        import numpy as np
        return np.vstack([self.phi1, self.PHI, self.phi2]).T

#-------------------------------------------------------------------------------

    def to_nfft(self):
        return matthies.from_bunge(self).to_nfft()
   
#-------------------------------------------------------------------------------
    
    @classmethod
    def from_array(cls, arg):
        '''
        create bunge object from numpy array
        '''
        return cls(arg[:,0], arg[:,1], arg[:,2])
            
#-------------------------------------------------------------------------------

    @classmethod
    def from_angax(cls, arg):
        return bunge.from_rmat(rmat.from_angax(arg))

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_quat(cls, arg):
        return bunge.from_rmat(rmat.from_quat(arg))

#-------------------------------------------------------------------------------
        
    @classmethod
    def from_rodri(cls, arg):
        return bunge.from_angax(angax.from_rodri(arg))

#-------------------------------------------------------------------------------
        
    @classmethod
    def from_matthies(cls, arg):
        import cryspy.util as util
        import numpy as np
        
        ind = ~util.isnegligible(arg.beta)
        
        phi1 = arg.alpha        
        phi1[ind] = phi1[ind] + np.pi / 2.0
        
        phi2 = arg.gamma
        phi2[ind] = phi2[ind] + 3.0 * np.pi / 2.0
        
        return bunge(phi1, arg.beta, phi2)

#-------------------------------------------------------------------------------

    @classmethod
    def from_rmat(cls, arg):
        import numpy as np
        
        # Make sure the g33 values are in range
        c2 = np.amax( \
                np.vstack([np.amin( \
                                np.vstack([arg.g33, 
                                           np.tile(1.0, arg.g33.shape[0])]).T,\
                                           axis=1), 
                                np.tile(-1.0, arg.g33.shape[0])]).T, axis=1)
        p  = np.absolute(c2) > 1.0 - np.sqrt(np.spacing(1))
        
        s2i = 1.0 / np.sqrt(1.0 - c2**2.0)
        c1  = -arg.g32 * s2i
        s1  =  arg.g31 * s2i
        c3  =  arg.g23 * s2i
        s3  =  arg.g13 * s2i
        
        # fix points near poles
        c1[p] = arg.g11[p]
        s1[p] = arg.g12[p]
        c3[p] = 1.0
        s3[p] = 0.0

        # phi1, PHI, phi2
        p1 = np.arctan2(s1, c1)
        P0 = np.arccos(c2)
        p2 = np.arctan2(s3, c3)
        
        # positive convention
        p1[p1<0] = p1[p1<0] + 2.0 * np.pi
        P0[P0<0] = P0[P0<0] + 2.0 * np.pi
        p2[p2<0] = p2[p2<0] + 2.0 * np.pi
        
        return cls(phi1=p1, PHI=P0, phi2=p2)   

#-------------------------------------------------------------------------------

    def __repr__(self):
        
        import numpy as np
        

        repstr = '{0: ^29s}\n'.format('Bunge Euler angles (\xb0)') + \
                 '{0:-^29s}\n'.format('-') + \
                 '  {0:^7s}, {1:^7s}, {2:^7s}  \n'.format('phi1',
                                                          'PHI',
                                                          'phi2')
        fmtstr = '[ {0: >7.3f}, {1: >7.3f}, {2: >7.3f} ]\n'

        n = self.size - 1
        if n > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.phi1[i] * 180.0/np.pi, 
                                         self.PHI[i] * 180.0/np.pi, 
                                        self.phi2[i] * 180.0/np.pi)
            for i in np.arange(0, 3):
                repstr += '  {0:^7s}{0:^7s}{0:^7s}  \n'.format('.')

            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.phi1[i] * 180.0/np.pi, 
                                         self.PHI[i] * 180.0/np.pi, 
                                        self.phi2[i] * 180.0/np.pi)
        else:
            
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.phi1[i] * 180.0/np.pi, 
                                         self.PHI[i] * 180.0/np.pi, 
                                        self.phi2[i] * 180.0/np.pi)
                                        
        repstr += '\n'
        return repstr

#-------------------------------------------------------------------------------

    def __getitem__(self,index):
        p1 = self.phi1[index]
        P0 = self.PHI[index]
        p2 = self.phi2[index]
        return bunge(phi1=p1, PHI=P0, phi2=p2)

#-------------------------------------------------------------------------------
class quat(object):
    ''' quaternion class

    Quaternion rotation representation.
    
    ...
    
    Attributes
    ----------
    a : scalar, list, tuple, or numpy array of floats
        the scalar part of the quaternion
    b  : scalar, list, tuple, or numpy array of floats
        the first value of the vector part of the quaternion
    c  : scalar, list, tuple, or numpy array of floats
        the second value of the vector part of the quaternion
    d  : scalar, list, tuple, or numpy array of floats
        the third value of the vector part of the quaternion
    
    Notes
    -----
    - The quaternion parameters (a, b, c, d) are normalized during
       initialization to produce a unit quaternion.
    
    - Quaternions cannot be used to represent improper rotations.
    
    - The convention is employed that convention that the scalar part of the 
      quaternion cannot take on negative values. The vector part is 
      automatically adjusted accordingly during initialization.
    
    - Coded with some inspiration from the robotics toolbox quaternion class by 
      Luis Fernando Lara Tobar and Peter Corke (governed by the Mozilla Public
      License 1.1) and the mtex toolbox by R. Hielscher and H. Schaeben 
      (governed by GPL2), and the matlab and octave functions for computer
      vision and image processing from Peter Kovesi.
    
    Examples
    --------
    >>> q = quat()
    
    >>> q = quat(0.6070,-0.7043,0.0634,-0.3627)
    
    >>> a = 0.6070,0.9688
    >>> b = -0.7043,0.1176
    >>> c = 0.0634,0.0263
    >>> d = -0.3627,0.2166
    >>> q = quat(a, b, c, d)
    
    >>> a = [0.6070,0.9688]
    >>> b = [-0.7043,0.1176]
    >>> c = [0.0634,0.0263]
    >>> d = [-0.3627,0.2166]
    >>> q = quat(a, b, c, d)
    
    >>> import numpy as np
    >>> a = np.array([0.6070,0.9688])
    >>> b = np.array([-0.7043,0.1176])
    >>> c = np.array([0.0634,0.0263])
    >>> d = np.array([-0.3627,0.2166])
    >>> q = quat(a, b, c, d)
    
    >>> b = bunge(np.pi, np.pi/2, np.pi/7)
    >>> q = quat.from_bunge(b)
    '''
    
    def __init__(self, a=1, b=0, c=0, d=0):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type a=..., b=... etc.
        import numpy as np
        import cryspy.util as util
        
        # check that the shapes of all are the same
        if np.shape(a)==np.shape(b)==np.shape(c)==np.shape(d):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            a = util.vecarrayconvert(a)
            b = util.vecarrayconvert(b)
            c = util.vecarrayconvert(c)
            d = util.vecarrayconvert(d)

            nrm=np.array([a, b, c, d])
            nrm=1.0 / np.sum(nrm**2.0, axis=0)

            self.a = a * nrm
            self.b = b * nrm
            self.c = c * nrm
            self.d = d * nrm   
            self.size = np.size(self.a)
        
        else:
            print "quat construction error: check that the lengths of a, b, "\
                  "c, and d are all the same."
            return None 

#-------------------------------------------------------------------------------

    def numel(self):
        ''' redundant with self.size
        '''
        return self.a.shape[0]

#-------------------------------------------------------------------------------

    def to_array(self):
        ''' convert quaternion object to numpy array
        '''
        import numpy as np
        return np.vstack([self.a, self.b, self.c, self.d]).T

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_array(cls,arg):
        ''' create quaternion object from numpy array
        '''
        return quat(arg[:,0], arg[:,1], arg[:,2], arg[:,3])
      
#-------------------------------------------------------------------------------
    
    @classmethod
    def from_angax(cls,arg):
        ''' convert angax to quat
        
        Conversion of angle/axis rotation representation to quaternion rotation
        representation
        
        Notes
        -----        
        Follows algorithm from Peter Kovesi (pk at csse uwa edu au), School of 
        Computer Science & Software Engineering, The University of Western
        Australia, http://www.csse.uwa.edu.au/
        '''
        import numpy as np
        an = 1.0 / np.sqrt(arg.x**2 + arg.y**2 + arg.z**2)
        x  = -arg.x * an
        y  = -arg.y * an
        z  = -arg.z * an
        hth = 0.5 * arg.th
        sth = np.sin(hth)        
        
        return cls(a=np.cos(0.5 * hth), b=x*sth, c=y*sth, d=z*sth)

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_bunge(cls,arg):
        ''' convert bunge to quat
        
        Conversion of Bunge Euler angle rotation representation to quaternion
        rotation representation.
        
        Notes
        -----
        - follows an algorithm similar to the one used in mtex; however,      
          gamma in mtex is phi2-2*pi/2. Here we use phi2+pi/2, which
          agrees with the Kocks chapter in Texture and Anisotropy.
        '''
        import numpy as np
        
        ha = 0.5 * (arg.phi1 - np.pi/2) # half alpha
        hb = 0.5 *  arg.PHI # half beta
        hg = 0.5 * (arg.phi2 + np.pi/2) # half gamma
        zz = np.zeros(np.size(ha)) # zeros array
        
        qa = quat(a=np.cos(ha), b=zz, c=zz,      d=np.sin(ha))
        qb = quat(a=np.cos(hb), b=zz, c=np.sin(hb), d=zz     )
        qc = quat(a=np.cos(hg), b=zz, c=zz,      d=np.sin(hg))
        
        return quat.conj(qa * qb * qc)

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_rodri(cls,arg):
        ''' convert rodri to quat
        
        Conversion of Rodrigues vector rotation representation to quaternion
        rotation representation.
        '''
        return quat.from_angax(angax.from_rodri(arg))

#-------------------------------------------------------------------------------

    @classmethod
    def from_rmat(cls,arg):
        ''' convert rmat to quat
        
        Conversion of rotation matrices to quaternion rotation representation.
        
        Notes
        -----
        - What happens when the rotation matrix describes an improper rotation?
          Quaternions can only handle proper rotations...
        - Follows an algorithm derived from the one used in mtex
        '''
        import numpy as np

        uu = arg.g11
        vv = arg.g22
        ww = arg.g33
        
        dd = np.zeros([np.size(uu), 4])
        dd[:, 0] = np.sqrt(1.0 + uu + vv + ww)
        dd[:, 1] = np.sqrt(1.0 - uu - vv + ww)
        dd[:, 2] = np.sqrt(1.0 + uu - vv - ww)
        dd[:, 3] = np.sqrt(1.0 - uu + vv - ww)
        
        # maximum values and locations in dd
        mh = np.amax(dd, axis=1)
        j  = np.argmax(dd, axis=1)
        
        mh = 0.50 * mh # convenience variable
        mi = 0.25 / mh # convenience variable
        
        n = np.size(arg.g11)
        
        a = np.zeros(n)
        b = np.copy(a)
        c = np.copy(a)
        d = np.copy(a)
        
        if n>1:
            
            x = j==0
            a[x] =  -mh[x]
            b[x] = ( arg.g23[x] - arg.g32[x]) * mi[x]
            c[x] = ( arg.g31[x] - arg.g13[x]) * mi[x]
            d[x] = ( arg.g12[x] - arg.g21[x]) * mi[x]
            
            x = j==1
            d[x] =   mh[x]
            c[x] = ( arg.g23[x] + arg.g32[x]) * mi[x]
            b[x] = ( arg.g31[x] + arg.g13[x]) * mi[x]
            a[x] =-( arg.g12[x] - arg.g21[x]) * mi[x]
            
            x = j==2
            b[x] =   mh[x]
            a[x] = (-arg.g23[x] + arg.g32[x]) * mi[x]
            d[x] = ( arg.g31[x] + arg.g13[x]) * mi[x]
            c[x] = ( arg.g12[x] + arg.g21[x]) * mi[x]
            
            x = j==3
            c[x] =   mh[x]
            d[x] = ( arg.g23[x] + arg.g32[x]) * mi[x]
            a[x] = (-arg.g31[x] + arg.g13[x]) * mi[x]
            b[x] = ( arg.g12[x] + arg.g21[x]) * mi[x]
            
            loc = a<0.0
            a[loc] = -a[loc]
            b[loc] = -b[loc]
            c[loc] = -c[loc]
            d[loc] = -d[loc]
        
        else:
        
            if j==0:
                a =  -mh
                b = ( arg.g23 - arg.g32) * mi
                c = ( arg.g31 - arg.g13) * mi
                d = ( arg.g12 - arg.g21) * mi
            
            if j==1:
                d =   mh
                c = ( arg.g23 + arg.g32) * mi
                b = ( arg.g31 + arg.g13) * mi
                a =-( arg.g12 - arg.g21) * mi
            
            if j==2:
                b =   mh
                a = (-arg.g23 + arg.g32) * mi
                d = ( arg.g31 + arg.g13) * mi
                c = ( arg.g12 + arg.g21) * mi
            
            if j==3:
                c =   mh
                d = ( arg.g23 + arg.g32) * mi
                a = (-arg.g31 + arg.g13) * mi
                b = ( arg.g12 + arg.g21) * mi
            
            if a<0.0:
                a = -a
                b = -b
                c = -c
                d = -d
            
            a = np.squeeze(a)
            b = np.squeeze(b)
            c = np.squeeze(c)
            d = np.squeeze(d)
        
        return quat(a, b, c, d)

#-------------------------------------------------------------------------------

    def __repr__(self):
        
        import numpy as np
        repstr = '{0: ^31s}\n'.format('quaternions') + \
                 '{0:-^31s}\n'.format('-') + \
                 ' {0:^6s} {1:^6s} {2:^6s} {3:^6s}  \n'.format('a', 'b',
                                                             'c', 'd')
        fmtstr = '<{0: >6.3f} {1: >6.3f} {2: >6.3f} {3: >6.3f} >\n'
        
        n = self.size - 1

        if self.size > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.a[i], 
                                        self.b[i], 
                                        self.c[i], 
                                        self.d[i])
            for i in np.arange(0, 3):
                repstr += ' {0:^6s} {0:^6s} {0:^6s} {0:^6s}  \n'.format('.')
            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.a[i], 
                                        self.b[i], 
                                        self.c[i], 
                                        self.d[i])  

        else:
            a = np.atleast_1d(self.a.squeeze())
            b = np.atleast_1d(self.b.squeeze())
            c = np.atleast_1d(self.c.squeeze())
            d = np.atleast_1d(self.d.squeeze())
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(a[i], b[i], c[i], d[i])
                                        
        repstr += '\n'
        return repstr

#------------------------------------------------------------------------------

    def __mul__(self, q2):
        ''' product
        
        Multiplication of quaternions
        
        Notes
        -----
        q1 and q2 must be the same size or one must be single.
        '''
        if isinstance(q2, quat):
            # the following works if q1 and q2 are the same size or one of them 
            # is a single quaternion. It rightfully returns an error otherwise.

            a = self.a * q2.a - self.b * q2.b - self.c * q2.c - self.d * q2.d
            b = self.a * q2.b + self.b * q2.a + self.c * q2.d - self.d * q2.c
            c = self.a * q2.c + self.c * q2.a + self.d * q2.b - self.b * q2.d
            d = self.a * q2.d + self.d * q2.a + self.b * q2.c - self.c * q2.b
            
            loc = a < 0.0
            a[loc] = -a[loc]
            b[loc] = -b[loc]
            c[loc] = -c[loc]
            d[loc] = -d[loc]
            
            return quat(a, b, c, d)

#------------------------------------------------------------------------------        
        
    def __truediv__(self, q):
        '''Returns quaternion misorientation.'''
        
        if isinstance(q, quat):
            qr = quat()
            qr = self * q.inv()    
        return qr

#------------------------------------------------------------------------------

    def inv(self):
        import copy
        '''Returns the quaternion inverse.'''
        qi = copy.copy(self)
        qi.a = -qi.a;
        
        return qi
            
#------------------------------------------------------------------------------

    def conj(self):
        ''' quaternion conjugate
        
        Returns the conjugate rotation for the quaternion
        
        '''
        import copy
        qi = copy.copy(self)
        qi.b = -qi.b
        qi.c = -qi.c
        qi.d = -qi.d
        
        return qi
            
#------------------------------------------------------------------------------
            
    def __getitem__(self,index):
        
        aa = self.a[index]
        bb = self.b[index]
        cc = self.c[index]
        dd = self.d[index]
        
        return quat(a=aa, b=bb, c=cc, d=dd)

#------------------------------------------------------------------------------

    def symmetrize(self, s):
        ''' returns symmetrically equivalent rotations
        
        Parameters
        ----------
        s : cryspy.xtal.symm object or cryspy.rot.quat object
        
        Returns
        -------
        list :
            in which each entry is a symmetrically equivalent version
        
        Notes
        -----
        Quaternions can only handle proper rotations. The symmetrize method
        when applied to quaternion class may produce a different number
        of results than the symmetrize method applied to a different class.
        '''
        # active rotation by symmetry
        if isinstance(s, quat):
            return [self * item for item in s]
        else:
            return [self * item for item in s.rotations]

#------------------------------------------------------------------------------

    def to_fundzone(self, s, qref=None):
        '''
        projects quaternions to the quaternion fundamental zone for provided
        symmetry s
        '''
        import numpy as np
        
        if qref==None:
            qref=quat()
        
        s = s.rotations * qref
        
        # get the number of quaternions and the number of symmetries        
        m = s.size
        n = self.size
        
        # compute all distances to fundamental regions
        w = np.tile(s.a, [n, 1]).T * np.tile(self.a, [m, 1])
        x = np.tile(s.b, [n, 1]).T * np.tile(self.b, [m, 1])
        y = np.tile(s.c, [n, 1]).T * np.tile(self.c, [m, 1])
        z = np.tile(s.d, [n, 1]).T * np.tile(self.d, [m, 1])
        d = np.absolute(w + x + y + z)
        
        # locate and extract the fundamental region for each quaternion
        loc = np.argmax(d, axis=0)
        q2f = s[[loc]].conj()  
        
        # project to the fundamental region
        qf = q2f * self
        
        return qf

#------------------------------------------------------------------------------

    @classmethod
    def rand(cls, n, kappa=0, mu=None):
        """ random quaternion(s) generated from von Mises-Fisher distribution
        
        Parameters
        ----------
        n : int
            number of random quaternions
        
        kappa : float > 0
            clustering parameter
        
        mu : quat instance
            location parameter (quaternion central to the distribution)
        
        Returns
        -------
        qr : quat instance
            random quaternions
        
        See Also
        --------
        util.rand_vonMisesFisherM     
        """
        import cryspy.util as util
        if mu == None:
            mu = quat().to_array()
        elif isinstance(mu, quat):
            mu = mu.to_array()
        else:
            print 'quat.rand: mu must be an instance of the quat class'
            
        return quat.from_array(util.rand_vonMisesFisherM(n=n, 
                                                         kappa=kappa, 
                                                         mu=mu))

#------------------------------------------------------------------------------

class rodri(object):
    """
    Rodrigues vector class
    
    Created on Fri Apr 13 16:20:21 2012
    
    @author: epayton
    """
    
    def __init__(self, r1=0, r2=0, r3=0):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type r1=..., r2=... etc.
        import numpy as np
        import cryspy.util as util
        
        # check that the shapes of all are the same
        if np.shape(r1)==np.shape(r2)==np.shape(r3):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            r1 = util.vecarrayconvert(r1)
            r2 = util.vecarrayconvert(r2)
            r3 = util.vecarrayconvert(r3)

            self.r1 = r1
            self.r2 = r2
            self.r3 = r3   
            
            self.size = np.size(r1)
        
        else:
            print "rodri construction error: check that the lengths of r1, r2,"\
                  " and r3 are all the same."
            return None 

#-------------------------------------------------------------------------------

    def numel(self):
        return self.r1.shape[0]

#-------------------------------------------------------------------------------

    def to_array(self):
        import numpy as np
        return np.vstack([self.r1, self.r2, self.r3]).T
           
#-------------------------------------------------------------------------------
            
    @classmethod
    def from_angax(cls, arg):
        import numpy as np
        nrm = np.tan(0.5 * arg.th)
        return rodri(r1=arg.x*nrm, r2=arg.y*nrm, r3=arg.z*nrm)

#-------------------------------------------------------------------------------

    @classmethod
    def from_bunge(cls, arg):
        return rodri.from_angax(angax.from_bunge(arg))

#-------------------------------------------------------------------------------

    @classmethod
    def from_quat(cls, arg):
        return rodri.from_angax(angax.from_quat(arg))

#-------------------------------------------------------------------------------

    @classmethod
    def from_rmat(cls, arg):
        return rodri.from_angax(angax.from_rmat(arg))

#-------------------------------------------------------------------------------

    def __repr__(self):
        
        import numpy as np
        repstr = '{0:^24s}\n'.format('Rodrigues vectors') + \
                 '{0:-^24s}\n'.format('-') + \
                 '  {0: ^6s} {1: ^6s} {2: ^6s}  \n'.format('r1', 'r2', 'r3')
        fmtstr = '< {0: >6.3f} {1: >6.3f} {2: >6.3f} >\n'
        
        n = self.size - 1

        if self.size > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.r1[i], 
                                        self.r2[i], 
                                        self.r3[i])
            for i in np.arange(0, 3):
                repstr += '  {0:^6s} {0:^6s} {0:^6s}  \n'.format('.')
            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.r1[i], 
                                        self.r2[i], 
                                        self.r3[i])  

        else:
  
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.r1[i], 
                                        self.r2[i], 
                                        self.r3[i])
                                        
        repstr += '\n'
        return repstr

#-------------------------------------------------------------------------------
class rmat(object): 
    ''' rotation matrix class
    
    object for containing and manipulating rotation matrices
    '''

    def __init__(self, g11=1, g12=0, g13=0, g21=0, g22=1, g23=0, g31=0, g32=0, g33=1):
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type g11=..., g2=... etc.
        import numpy as np
        import cryspy.util as util
        
        # check that the shapes of all are the same
        if np.shape(g11)==np.shape(g12)==np.shape(g13)==\
           np.shape(g21)==np.shape(g22)==np.shape(g23)==\
           np.shape(g31)==np.shape(g32)==np.shape(g33):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            self.g11 = util.vecarrayconvert(g11)
            self.g12 = util.vecarrayconvert(g12)
            self.g13 = util.vecarrayconvert(g13)
            self.g21 = util.vecarrayconvert(g21)
            self.g22 = util.vecarrayconvert(g22)
            self.g23 = util.vecarrayconvert(g23)
            self.g31 = util.vecarrayconvert(g31)
            self.g32 = util.vecarrayconvert(g32)
            self.g33 = util.vecarrayconvert(g33)
            
            self.size = np.size(g11)
        
        else:
            print "rmat construction error: check that the lengths of g11, g12,"\
                  " ... g33 are all the same."
            return None  

#------------------------------------------------------------------------------

    def numel(self):
        """ returns rmat size
        
        Gives the number of elements in the rotation matrix object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        n : int
            the number of elements in the rotation matrix object
        
        Examples
        --------
        >>> import cryspy.rot as rot
        >>> my_rmat = rot.rmat(g1=[1,-1], g2=[0, 0], g3=[0, 0],
        ...                    g21=[0, 0], g22=[1,-1], g23=[0, 0],
        ...                    g31=[0, 0], g32=[0, 0], g33=[1,-1])
        >>> numel = my_rmat.numel()
        """
        return self.g11.shape[0]

#------------------------------------------------------------------------------

    def to_array(self):
        """ convert rmat to array
        
        Converts a rotation matrix object to an n x 9 numpy array.
        
        Examples
        --------
        >>> import cryspy.rot as rot
        >>> my_rmat = rot.rmat(1, 0, 0, 0, 1, 0, 0, 0, 1)
        >>> my_array = my_rmat.to_array()
        """
        import numpy as np
        return np.vstack([self.g11, self.g12, self.g13, 
                          self.g21, self.g22, self.g23,
                          self.g31, self.g32, self.g33]).T
         
#------------------------------------------------------------------------------


    @classmethod
    def from_array(cls,arg):
        """ convert array to rmat
        
        Class method for converting an n x 9 numpy array into an object of the
        rotation matrix class.
        
        Parameters
        ----------
        arg : n x 9 numpy array of floats
            the numpy array from which to create the rmat
        
        Examples
        --------
        >>> import numpy as np
        >>> import cryspy.rot as rot
        >>> my_array = np.array([[ 1, 0, 0, 0, 1, 0, 0, 0, 1],
        ...                      [-1, 0, 0, 0,-1, 0, 0, 0,-1]])
        >>> my_rmats = rot.rmat.from_array( my_array )
        """
        import numpy as np

        s = np.shape(arg)
        if np.all(np.asarray(s) != 9):
            print 'error: rotation object requires at least one dimension = 9'
        
        if np.size(s) == 1:
            return cls(g11=arg[0], g12=arg[1], g13=arg[2], \
                       g21=arg[3], g22=arg[4], g23=arg[5], \
                       g31=arg[6], g32=arg[7], g33=arg[8])
        else:
            return cls(g11=arg[:,0], g12=arg[:,1], g13=arg[:,2], \
                       g21=arg[:,3], g22=arg[:,4], g23=arg[:,5], \
                       g31=arg[:,6], g32=arg[:,7], g33=arg[:,8])            
                

#------------------------------------------------------------------------------

    @classmethod
    def from_quat(cls,arg):
        """ convert quat to rmat
        
        Convert quaternion object to rotation matrix object
        
        Examples
        --------
        >>> import cryspy.rot as rot
        >>> my_quat_obj = rot.quat()
        >>> my_rmat = rot.rmat.from_quat( my_quat_obj )
        """
        
        # user wants to convert quaternions to rotation matrices
        
        w = arg.a
        x = arg.b
        y = arg.c
        z = arg.d;

        w2 = w**2.0
        x2 = x**2.0
        y2 = y**2.0
        z2 = z**2.0
        
        xy = x*y
        xz = x*z
        yz = y*z
        wx = w*x
        wy = w*y
        wz = w*z
        
        return cls(
            g11 = w2 + x2 - y2 - z2,
            g12 = 2.0 * (xy - wz),
            g13 = 2.0 * (xz + wy),
            g21 = 2.0 * (xy + wz),
            g22 = w2 - x2 + y2 - z2,
            g23 = 2.0 * (yz - wx),
            g31 = 2.0 * (xz - wy),
            g32 = 2.0 * (yz + wx),
            g33 = w2 - x2 - y2 + z2)

#------------------------------------------------------------------------------
    
    @classmethod
    def from_bunge(cls,arg):    
        # user wants to convert Bunge Euler angles to rotation matrices
        # algorithm from wikipedia
        import numpy as np        
        
        c1 = np.cos(arg.phi1)
        c2 = np.cos(arg.PHI)
        c3 = np.cos(arg.phi2)
        s1 = np.sin(arg.phi1)
        s2 = np.sin(arg.PHI)
        s3 = np.sin(arg.phi2)
        
        # ZXZ combination
        return cls(
            g11 =  c1 * c3 - s1 * c2 * s3,
            g12 = -s3 * c1 - c2 * c3 * s1,
            g13 =  s2 * s1,
            g22 = -s1 * s3 + c1 * c2 * c3,
            g21 =  c3 * s1 + s3 * c2 * c1,
            g23 = -s2 * c1,            
            g31 =  s3 * s2,
            g32 =  c3 * s2,
            g33 =  c2)

#------------------------------------------------------------------------------
            
    @classmethod
    def from_angax(cls,arg):
        """ Create rmat from angax
        
        Converts an angle/axis to a passive rotation matrix
        
        Parameters
        ----------
        arg : angax object
            object containin angle/axis pair
        
        Returns
        -------
        rmat : rmat object
            rotation matrix corresponding to angle/axis input
        
        Notes
        -----
        This converts an angle/axis rotation to a passive rotation matrix.        
        """
        import numpy as np
        r1 = arg.x
        r2 = arg.y
        r3 = arg.z
        r1s = r1**2.0
        r2s = r2**2.0
        r3s = r3**2.0
        r12 = r1 * r2
        r13 = r1 * r3
        r23 = r2 * r3
        c = np.cos(arg.th)
        s = np.sin(arg.th)
        cm = 1 - c
        
        return cls(
            g11 = r1s * cm + c,
            g12 = r12 * cm - r3 * s,
            g13 = r13 * cm + r2 * s,
            g21 = r12 * cm + r3 * s,
            g22 = r2s * cm + c,
            g23 = r23 * cm - r1 * s,
            g31 = r13 * cm - r2 * s,
            g32 = r23 * cm + r1 * s,
            g33 = r3s * cm + c)

#------------------------------------------------------------------------------
        
    @classmethod
    def from_rodri(cls,arg):
        return cls.from_angax(angax.from_rodri(arg))    

#------------------------------------------------------------------------------

    def __repr__(self):
        
        import numpy as np
        repstr = '{0:^72s}\n'.format('rotation matrices') + \
                 '{0:-^72s}\n'.format('-') + \
                 ' {0:^6s}  {1:^6s}  {2:^6s} '.format('g11', 'g12', 'g13') + \
                 ' {0:^6s}  {1:^6s}  {2:^6s} '.format('g21', 'g22', 'g23') + \
                 ' {0:^6s}  {1:^6s}  {2:^6s} \n'.format('g31', 'g32', 'g33')
                 
        fmtstr = '[{0: >6.3f}, {1: >6.3f}, {2: >6.3f};' + \
                 ' {3: >6.3f}, {4: >6.3f}, {5: >6.3f};' + \
                 ' {6: >6.3f}, {7: >6.3f}, {8: >6.3f}]\n'
        
        n = self.size - 1

        if self.size > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.g11[i], 
                                        self.g12[i], 
                                        self.g13[i], 
                                        self.g21[i], 
                                        self.g22[i],
                                        self.g23[i],
                                        self.g31[i],
                                        self.g32[i],
                                        self.g33[i])
            for i in np.arange(0, 3):
                repstr += ' {0:^6s}  {0:^6s}  {0:^6s} '.format('.') + \
                          ' {0:^6s}  {0:^6s}  {0:^6s} '.format('.') + \
                          ' {0:^6s}  {0:^6s}  {0:^6s} \n'.format('.')
            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.g11[i], 
                                        self.g12[i], 
                                        self.g13[i], 
                                        self.g21[i], 
                                        self.g22[i],
                                        self.g23[i],
                                        self.g31[i],
                                        self.g32[i],
                                        self.g33[i]) 

        else:
  
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.g11[i], 
                                        self.g12[i], 
                                        self.g13[i], 
                                        self.g21[i], 
                                        self.g22[i],
                                        self.g23[i],
                                        self.g31[i],
                                        self.g32[i],
                                        self.g33[i])
                                        
        repstr += '\n'
        return repstr

#------------------------------------------------------------------------------
            
    def __getitem__(self, index):
        gg11 = self.g11[index]
        gg12 = self.g12[index]
        gg13 = self.g13[index]
        gg21 = self.g21[index]
        gg22 = self.g22[index]
        gg23 = self.g23[index]
        gg31 = self.g31[index]
        gg32 = self.g32[index]
        gg33 = self.g33[index]
        return rmat(g11=gg11, g12=gg12, g13=gg13,
                    g21=gg21, g22=gg22, g23=gg23,
                    g31=gg31, g32=gg32, g33=gg33)

#------------------------------------------------------------------------------

    def __mul__(self, b):
        """
        multiplies rotation matrices
        
        uses Laderman method (23 steps rather than 27)
        
        Reference
        ---------
        Laderman, JD. "A Noncommutative Algorithm for Multiplying 3×3 Matrices
        Using 23 Multiplications." Bull Amer Math Soc 82 (1976) p126. 
        """
        if isinstance(b, rmat):
            
           m1  = ( self.g11 + self.g12 + self.g13 - self.g21 - \
                   self.g22 - self.g32 - self.g33 ) * b.g22
           m2  = ( self.g11 - self.g21 ) * ( -b.g12 + b.g22 )
           m3  =   self.g22 * (-b.g11 + b.g12 + b.g21 - b.g22 - b.g23 - b.g31 + b.g33 )
           m4  = (-self.g11 + self.g21 + self.g22 ) * ( b.g11 - b.g12 + b.g22 )
           m5  = ( self.g21 + self.g22 ) * (-b.g11 + b.g12 )
           m6  =   self.g11 * b.g11
           m7  = (-self.g11 + self.g31 + self.g32 ) * ( b.g11 - b.g13 + b.g23 )
           m8  = (-self.g11 + self.g31 ) * ( b.g13 - b.g23 )
           m9  = ( self.g31 + self.g32 ) * (-b.g11 + b.g13 )
           m10 = ( self.g11 + self.g12 + self.g13 - self.g22 - \
                   self.g23 - self.g31 - self.g32 ) * b.g23
           m11 =   self.g32 * (-b.g11 + b.g13 + b.g21 - b.g22 - b.g23 - b.g31 + b.g32 )
           m12 = (-self.g13 + self.g32 + self.g33 ) * (b.g22 + b.g31 - b.g32 )
           m13 = ( self.g13 - self.g33) * ( b.g22 - b.g32)
           m14 =   self.g13 * b.g31
           m15 = ( self.g32 + self.g33 ) * (-b.g31 + b.g32 )
           m16 = (-self.g13 + self.g22 + self.g23 ) * ( b.g23 + b.g31 - b.g33)
           m17 = ( self.g13 - self.g23 ) * ( b.g23 - b.g33 )
           m18 = ( self.g22 + self.g23 ) * (-b.g31 + b.g33 )
           m19 =   self.g12 * b.g21
           m20 =   self.g23 * b.g32
           m21 =   self.g21 * b.g13
           m22 =   self.g31 * b.g12
           m23 =   self.g33 * b.g33
        
           c1 = m6  + m14 + m19
           c2 = m1  + m4  + m5  + m6  + m12 + m14 + m15
           c3 = m6  + m7  + m9  + m10 + m14 + m16 + m18
           c4 = m2  + m3  + m4  + m6  + m14 + m16 + m17
           c5 = m2  + m4  + m5  + m6  + m20
           c6 = m14 + m16 + m17 + m18 + m21
           c7 = m6  + m7  + m8  + m11 + m12 + m13 + m14
           c8 = m12 + m13 + m14 + m15 + m22
           c9 = m6  + m7  + m8  + m9  + m23
           
           return rmat(g11=c1, g12=c2, g13=c3, 
                       g21=c4, g22=c5, g23=c6, 
                       g31=c7, g32=c8, g33=c9)
        else:
            print 'rmat multiplication error'

#------------------------------------------------------------------------------

    def __truediv__(self, b):
        """
        matrix right division, a/b
        """
        if isinstance(b, rmat):        
            return self * b.inv()

#------------------------------------------------------------------------------

    def inv(self):
        """
        returns the inverses of the matrices
        """
        di = 1.0/self.det()
        badj = self.adj()
        c1 = badj.g11 * di
        c2 = badj.g12 * di
        c3 = badj.g13 * di
        c4 = badj.g21 * di
        c5 = badj.g22 * di
        c6 = badj.g23 * di
        c7 = badj.g31 * di
        c8 = badj.g32 * di
        c9 = badj.g33 * di     

        return rmat(g11=c1, g12=c2, g13=c3, 
                    g21=c4, g22=c5, g23=c6, 
                    g31=c7, g32=c8, g33=c9)
          
#------------------------------------------------------------------------------

    def det(self):
        """
        returns the determinants of the matrices
        """
        d = self.g11 * self.g22 * self.g33 - \
            self.g11 * self.g23 * self.g32 - \
            self.g12 * self.g21 * self.g33 + \
            self.g12 * self.g23 * self.g31 + \
            self.g13 * self.g21 * self.g32 - \
            self.g13 * self.g22 * self.g31
        return d

#------------------------------------------------------------------------------ 
        
    def adj(self):
        """
        returns the adjugate matrices
        """
        m1 =  self.g22 * self.g33 - self.g23 * self.g32
        m2 = -self.g12 * self.g33 + self.g13 * self.g32
        m3 =  self.g12 * self.g23 - self.g22 * self.g13
        m4 = -self.g21 * self.g33 + self.g23 * self.g31
        m5 =  self.g11 * self.g33 - self.g13 * self.g31
        m6 = -self.g11 * self.g23 + self.g13 * self.g21
        m7 =  self.g21 * self.g32 - self.g22 * self.g31
        m8 = -self.g11 * self.g32 + self.g12 * self.g31
        m9 =  self.g11 * self.g22 - self.g12 * self.g21

        return rmat(g11=m1, g12=m2, g13=m3, 
                    g21=m4, g22=m5, g23=m6, 
                    g31=m7, g32=m8, g33=m9)      

#------------------------------------------------------------------------------

    def trans(self):
        """
        returns the transpose
        """
        return rmat(g11=self.g11, g12=self.g21, g13=self.g31,
                    g21=self.g12, g22=self.g22, g23=self.g32,
                    g31=self.g13, g32=self.g23, g33=self.g33)

#------------------------------------------------------------------------------

    def trace(self):
        """
        returns the trace
        """
        import numpy as np
        return np.array(self.g11 + self.g22 + self.g33)


#------------------------------------------------------------------------------ 

class matthies(object):
    '''
    Matthies Euler angles rotation representation.
    
    ...
    
    Attributes
    ----------
    alpha : list, tuple, or numpy array of floats
        the angle phi1 describing the first rotation (about the x reference
        axis), in radians
    beta  : list, tuple, or numpy array of floats
        the angle PHI describing the second rotation (about the z reference
        axis), in radians
    gamma  : list, tuple, or numpy array of floats
        the angle phi2, describing the third rotation (about the x reference
        axis), in radians
    '''

    def __init__(self, alpha=0, beta=0, gamma=0):
        import numpy as np
        import cryspy.util as util
        # note that in the above initialization, the default values will be
        # be given for the remaining parameters if insufficient values are
        # passed to the function. The best way to avoid this problem is to
        # always type phi1=..., PHI=... etc.
        
        # check that the shapes of all are the same
        if np.shape(alpha)==np.shape(beta)==np.shape(gamma):      
        
            # the following may look ridiculous, but it allows the arguments to
            #  be passed as np.matrices, np.arrays, tuples, or lists...        
            self.alpha = util.vecarrayconvert(alpha)
            self.beta  = util.vecarrayconvert(beta)
            self.gamma = util.vecarrayconvert(gamma)
            self.size = np.size(self.alpha)
        
        else:
            print "bunge construction error: check that the lengths of phi1, "\
                  "PHI, and phi2 are all the same."
            return None
            
#-------------------------------------------------------------------------------

    def numel(self):
        return self.alpha.shape[0]

#-------------------------------------------------------------------------------

    def to_array(self):
        import numpy as np
        return np.vstack([self.alpha, self.beta, self.gamma]).T

#-------------------------------------------------------------------------------

    def to_nfft(self):
        ''' converts Matthies Euler angles into nfft convention
        
        Note
        ----
        - Algorithm from mtex 4.1.4
        '''
        import numpy as np
        
        def fft_rho(rho):
            # This is the contents of fft_rho.m
            rho = np.mod(rho + np.pi, 2.0 * np.pi) - np.pi
            prho = rho / 2.0 / np.pi
            prho[prho >= 0.5] = -0.5
            return np.array(prho)
        
        def fft_theta(th):
            # This is the contents of fft_theta.m
            th = np.mod(th, 2.0 * np.pi)
            pth = th / 2.0 / np.pi
            return np.array(pth)
        
        al_fft = fft_rho(self.alpha)
        be_fft = fft_rho(self.beta) # TODO: MTEX has an 'nfft_bug' preference flag. It is false in my distribution. Check in to why this is there.
        ga_fft = fft_rho(self.gamma)
        
        return 2.0 * np.pi * np.vstack([al_fft, be_fft, ga_fft]).T
                
#-------------------------------------------------------------------------------
    
    @classmethod
    def from_array(cls,arg):
        '''
        create bunge object from numpy array
        '''
        return cls(arg[:,0], arg[:,1], arg[:,2])
            
#-------------------------------------------------------------------------------

    @classmethod
    def from_angax(cls,arg):
        return matthies.from_rmat(rmat.from_angax(arg))

#-------------------------------------------------------------------------------
    
    @classmethod
    def from_quat(cls, arg):
        '''
        
        Example
        -------
        >>> import numpy as np
        >>> import cryspy.rot as rot
        >>> v = np.arange(0.0, 2.0*np.pi, np.pi/7.0)
        >>> q = rot.quat.from_bunge(v, v, v)
        >>> m = rot.matthies.from_quat(q)
        
        Notes
        -----
        - Algorithm from mtex 4.1.4
        '''
        import numpy as np
        import cryspy.util as util

        q = arg.conj() # TODO: This is different from MTEX, since we are using a passive convention. The conjugate is required to get the same results... but is that what we want?
        at1 = np.arctan2(q.d, q.a)
        at2 = np.arctan2(q.b, q.c)
        alpha = at1 - at2
        beta = 2.0 * np.arctan2(np.sqrt(q.b**2 + q.c**2), 
                                np.sqrt(q.a**2 + q.d**2))
        gamma = at1 + at2
        
        ind = util.isnegligible(beta, factor=10.0) # FIXME: 10x is somewhat arbitrary here. Can we do something better?
        alpha[ind] = 2.0 * np.arcsin(np.maximum(-1.0, 
                                              np.minimum(1.0, 
                                                         util.ssign(q.a[ind]) * q.d[ind])))
        gamma[ind] = 0.0
        
        alpha = np.mod(alpha, 2.0 * np.pi)
        gamma = np.mod(gamma, 2.0 * np.pi)
        
        return matthies(alpha, beta, gamma)

#-------------------------------------------------------------------------------

    @classmethod
    def from_rodri(cls, arg):
        return matthies.from_quat(quat.from_rodri(arg))

#-------------------------------------------------------------------------------

    @classmethod
    def from_rmat(cls, arg):
        return matthies.from_quat(quat.from_rmat(arg)) 

#-------------------------------------------------------------------------------

    @classmethod
    def from_bunge(cls, arg):
        
        # algorithm from mtex 4.1.2
        import cryspy.util as util
        import numpy as np
        
        ind = ~util.isnegligible(arg.PHI)
        
        alpha = arg.phi1
        alpha[ind] = alpha[ind] - np.pi / 2.0
        
        gamma = arg.phi2
        gamma[ind] = gamma[ind] - 3.0 * np.pi / 2.0
        
        return matthies(alpha, arg.PHI, gamma)

#-------------------------------------------------------------------------------

    def __repr__(self):
        
        import numpy as np

        repstr = '{0:^29s}\n'.format('Matthies Euler angles (\xb0)') + \
                 '{0:-^29s}\n'.format('-') + \
                 '  {0:^7s}, {1:^7s}, {2:^7s}  \n'.format('alpha',
                                                          'beta',
                                                          'gamma')
        fmtstr = '[ {0: >7.3f}, {1: >7.3f}, {2: >7.3f} ]\n'

        n = self.size - 1
        if n > 15:
            
            for i in np.arange(0, 5):
                repstr += fmtstr.format(self.alpha[i] * 180.0/np.pi, 
                                         self.beta[i] * 180.0/np.pi, 
                                        self.gamma[i] * 180.0/np.pi)
            for i in np.arange(0, 3):
                repstr += '  {0:^7s}  {0:^7s}  {0:^7s}  \n'.format('.')

            for i in np.arange(n-5, n):
                repstr += fmtstr.format(self.alpha[i] * 180.0/np.pi, 
                                         self.beta[i] * 180.0/np.pi, 
                                        self.gamma[i] * 180.0/np.pi)
        else:
            
            for i in np.arange(0, self.size):
                repstr += fmtstr.format(self.alpha[i] * 180.0/np.pi, 
                                         self.beta[i] * 180.0/np.pi, 
                                        self.gamma[i] * 180.0/np.pi)
                                        
        repstr += '\n'
        return repstr

#-------------------------------------------------------------------------------

    def __getitem__(self,index):
        
        alf = self.alpha[index]
        bet = self.beta[index]
        gam = self.gamma[index]
        
        return matthies(alpha=alf, beta=bet, gamma=gam)
    
#------------------------------------------------------------------------------        

def consolidate(list_of_rotation_objects):
    """ stack lists of rotation objects into numpy array
    
    Parameters
    ----------
    list_of_rotation_objects : list
        Python list containing cryspy.rot classes of the same type
    
    Returns
    -------
    numpy array 
    
    Notes
    -----
    To re-create a cryspy.rot class object with the results of consolidate, see
    the .from_array method for any of the classes.
    
    Examples
    --------
    >>> import cryspy.rot as rot
    >>> import cryspy.xtal as xtal
    >>> cs = xtal.symm('m-3m')
    >>> r = rot.quat.rand(10)
    >>> rs = r.symmetrize(cs)
    >>> symm_equivs_to_r = rot.consolidate(rs)
    >>> symm_equivs_as_quat = rot.quat.from_array(symm_equivs_to_r)
    """
    import numpy as np
    
    r = list_of_rotation_objects[0].to_array()
    
    if np.size(list_of_rotation_objects) > 1:
        
        return [np.vstack([r, item.to_array()]) for item in \
                                               list_of_rotation_objects[1:]][0]
    else:
        
        return r

#------------------------------------------------------------------------------        

# TODO: see if there is another, more elegant way to prevent import loops

### PREVENT IMPORT LOOPS ###
# ovlib.xtal depends on ovlib.rot.quat and ovlib.rot.rmat already being defined
from cryspy.xtal import miller