# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:20:55 2022

ODF Testbed

"""

from orix.quaternion import *
from orix.quaternion import symmetry


class ODF:
    """ Orientation distirbution docstring"""


    def __init__(self, bandwidth=32, ori_size = 1e6):
        """
        Generate initial plan, equivalent to setting plan in MTEX

        Parameters
        ----------
        bandwidth : int, optional
            Sets the bandwidth (ie, l) with which the ODF is calculated.
            A higher bandwidth increases accuracy and compute time at a 
            geometric rate. Default is 32

        ori_size : int, optional
            The max number of input orientations an ODF can be created from.
            Larger values will create larger ODFs in memory. 
            Default is 1 million

        Returns
        -------
        None.

        """
        # do the init suff here
#        plan = nfsoftmex('init',nfsoft_bandwidth,ori_size,nfsoft_flags,0,4,1000,nfsoft_size);

    def calc(g, kernel=Kernel.DVP(5), weights = 1):
        """
        

        Parameters
        ----------
        g : int, optional
            input orientations from which to build the ODF
        kernel : TYPE, optional
            DESCRIPTION. The default is Kernel.DVP(5).
        weights : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        #nfsoftmex('set_x',plan,Euler(g,'nfft').');
        #nfsoftmex('set_f',plan,c(:));
        #nfsoftmex('precompute',plan);
        #nfsoftmex('adjoint',plan);
        #c_hat = nfsoftmex('get_f_hat',plan);
        #nfsoftmex('finalize',plan);

        return()
    
    def evaluate(query):
        
        return()

    def __finalize__():
        go_kill_NFFT = True


    @classmethod
    def uniform()
        return(# Some ODF where everything returns 1)

    def __add__
        return(# Some ODF where everything returns 1)
               

    # example of what a classmethod might look like
    @classmethod
    def from_neo_euler(cls, neo_euler):
        """Creates a rotation from a neo-euler (vector) representation.

        Parameters
        ----------
        neo_euler : NeoEuler
            Vector parametrization of a rotation.
        """
        s = np.sin(neo_euler.angle / 2)
        a = np.cos(neo_euler.angle / 2)
        b = s * neo_euler.axis.x
        c = s * neo_euler.axis.y
        d = s * neo_euler.axis.z
        r = cls(np.stack([a, b, c, d], axis=-1))
        return r

# # Notes, not actual code
# """
# g c A 2 4 A

# A = forier coefficients for kernel (calculated during calc)
# c = weights for each orientation g
# g = input data orientations
# """



# nfsoft_flags = 2^4;
# % init
# plan = nfsoftmex('init',length(A)-1,length(g),nfsoft_flags,0,4,1000,2*ceil(1.5*(length(A)+1)));

# % Calc
# nfsoftmex('set_x',plan,Euler(g,'nfft').');
# nfsoftmex('set_f',plan,c(:));
# nfsoftmex('precompute',plan);
# nfsoftmex('adjoint',plan);
# c_hat = nfsoftmex('get_f_hat',plan);
# nfsoftmex('finalize',plan);

# % Add Kernel
# for l = 1:length(A)-1
#   ind = (deg2dim(l)+1):deg2dim(l+1);
#   c_hat(ind) = A(l+1)* reshape(c_hat(ind),2*l+1,2*l+1);
# end


# class Misorientation(Rotation):
#     r"""Misorientation object.

#     Misorientations represent transformations from one orientation,
#     :math:`o_1` to another, :math:`o_2`: :math:`o_2 \cdot o_1^{-1}`.

#     They have symmetries associated with each of the starting
#     orientations.
#     """

#     _symmetry = (C1, C1)

#     def __init__(self, data, symmetry=None):
#         super().__init__(data)
#         if symmetry:
#             self.symmetry = symmetry

#     @property
#     def symmetry(self):
#         """Tuple of :class:`~orix.quaternion.Symmetry`."""
#         return self._symmetry
