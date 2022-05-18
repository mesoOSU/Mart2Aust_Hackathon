# -*- coding: utf-8 -*-
# Created on Thu Feb 02 15:54:49 2017 by paytonej

class kernel:
    """ creates a kernel for odf functions        
    
    Notes
    -----
    - This is essentially a partial translation of the kernel class from MTEX
      4.1
    - So far only the deLaValeePoussin kernel has been implemented    
    """
    
    def __init__(self, name='deLaValeePoussin', halfwidth=10.0, psi=None, bandwidth=None, antipodal=True):
        
        import numpy as np
        import scipy.special as spec
        
        if name is 'deLaValeePoussin':
            kappa = 0.5 * np.log(0.5) / np.log(np.cos(halfwidth * np.pi / 360.0))
            bandwidth = np.round(kappa)
            
            # Constant C
            cc = spec.beta(1.5, 0.5) / spec.beta(1.5, kappa + 0.5)
            
            # Legendre Coefficients
            aa = np.ones(bandwidth + 1)
            aa[1] = kappa / (kappa + 2.0)
            for l in np.arange(1, bandwidth):
                aa[l+1] = ((kappa - l + 1) * aa[l-1] - (2.*l + 1.) * aa[l]) / (kappa + l + 2)
            
            for l in np.arange(0, bandwidth+1):
                aa[l] = (2. * l + 1.0) * aa[l]
            
        # Cut A (aa). This is a private function within the mtex kernel class.
        # Here we will organize it into the initialization of the kernel.
        epsilon = 0.01 # in mtex, this is set by an mtex preference
        ind = np.where(np.abs(aa) <= np.amax([np.amin(np.abs(aa)),
                                                          epsilon]))[0][0]
        
        self.name = name
        self.halfwidth = halfwidth
        self.kappa = kappa
        self.L = bandwidth
        self.C = cc
        self.A = aa[0:np.amin([ind + 1, np.shape(aa)[0]])] # Chebyshev coefficients
        self.antipodal = antipodal

    def __repr__(self):
        if self.antipodal == True:
            ant = 'antipodal'
        else:
            ant = ''
        print '{0}{1:s} kernel with halfwidth {1:.5f}\xb0'.format(ant, 
                                                               self.name, 
                                                               self.halfwidth)

#------------------------------------------------------------------------------

def deg2dim(l):
    """ return dimensions of the harmonic space up to order l
    
    Parameters
    ----------
    l : float
    
    Returns
    -------
    float
    """
    import numpy as np
    
    dim = l * (2.0 * l - 1.0) * (2.0 * l + 1.0) / 3.0
    
    return np.int(dim)

#------------------------------------------------------------------------------


# TODO: create similar functions to the below for fc2odf, odf2pf, pdf2pf, pf2odf. The code below should form a decent template to start from.
def odf2fc(quaternions, weights, kernel_legendre_coefficients):
    """ computes Fourier coefficients for ODF using MTEX algorithm
    
    Parameters
    ----------
    quaternions : cryspy.rot.quat class
        crystal rotations
    
    weights : 1 x n numpy array
        Weighting applied to each Euler angle
    
    kernel : cryspy.tx.kernel class
        Kernel function to be used in ODF calculation
    
    Returns
    -------
    1 x n numpy array
        Fourier coefficients
    
    Notes
    -----
    Performs an external function call to MTEX's ODF calculation functions.
    
    Reference
    ---------
    
    Examples
    --------
    >>> import cryspy.io as cio
    >>> import cryspy.tx as tx
    >>> import numpy as np
    >>> dat = cio.loadang('example_data.ang')
    >>> o = dat.orientations
    >>> c = c = np.ones(o.size)
    >>> k = tx.kernel()
    >>> fourier_coefficients = odf2fc(o.rotation, c, k)
    """
    import cryspy.rot as rot
    import cryspy.util as util
    import numpy as np
    import time
    import subprocess
    import io 

    nfft_euler = rot.matthies.from_quat(quaternions).to_nfft().T  
    
    # Convert our strings to byte string.
    gb = nfft_euler.astype(np.float64).tostring()
    cb = weights.astype(np.float64).tostring()
    aab = kernel_legendre_coefficients.astype(np.float64).tostring()
    
    prgname = 'odf2fc'
    
    h = util.helper() # helper contains our system architecture-specific paths
    prgpath = util.fullfile([h.prgpth, prgname + h.ext])
    
    # Give our temporary file a unique name using the time since epoch
    suffix = str(int(time.time() * 100.0))
    name = prgname + suffix
    
    # The listfile will point the external function to where the inputs have been
    # temporarily written. Each time we write a variable file, we will log it.
    listfilepath = util.fullfile([h.tmpdir, name + '.txt'])
    listfile = io.open(listfilepath, 'w', newline='') # XXX: newline='' prevents the \lf\cr line endings in Windows. It is not clear to me at this time that this will work in Unix or Linux.
    
    # Write g to temporary folder and log it
    datafilepath = util.fullfile([h.tmpdir, name + '_' + 'g' + '.dat'])
    with io.open(datafilepath, 'wb') as dfile:
        dfile.write(gb)
    listfile.write(unicode('g: ' + datafilepath + '\n', 'utf-8'))
    
    # Write c to temporary folder and log it
    datafilepath = util.fullfile([h.tmpdir, name + '_' + 'c' + '.dat'])
    with io.open(datafilepath, 'wb') as dfile:
        dfile.write(cb)
    listfile.write(unicode('c: ' + datafilepath + '\n', 'utf-8'))
    
    # Write A to temporary folder and log it
    datafilepath = util.fullfile([h.tmpdir, name + '_' + 'A' + '.dat'])
    with io.open(datafilepath, 'wb') as dfile:
        dfile.write(aab)
    listfile.write(unicode('A: ' + datafilepath + '\n', 'utf-8'))
    
    # Give a path for the output results file
    outputfilepath = util.fullfile([h.tmpdir, name + '_' + 'res1' + '.dat'])
    with io.open(datafilepath, 'wb') as dfile:
        dfile.write(aab)
    listfile.write(unicode('res1: ' + outputfilepath + '\n', 'utf-8'))
    
    listfile.close() # close the file listing inputs
    
    # Use subprocess to perform the external system call to our function.
    subprocess.call([prgpath,  listfilepath]) # TODO: place in try/except structure and return an error if function call is unsuccessful
    
    # remove the temporary file containing the list of program inputs
    del listfilepath
    
    # read in the output
    result = np.fromfile(outputfilepath, 'd')
    
    return result

#-----------------------------------------------------------------------------
 
def calcFourier(component, kernel, ss):
    
    import numpy as np
    import cryspy.rot as rot

    def gcA2Fourier(g, c, aa):
        """
        Get Fourier coefficients from g, c, A
        
        same as the following subfunction gcA2fourier in MTEX:
        [mtex dir]\ODFAnalysis\@unimodalComponent\calcFourier
        """
        import cryspy.tx as tx
        # this is a direct translation from mtex
        # run NFSOFT
        f = tx.odf2fc(g, c, aa)
        return f[0::2] + 1j * f[1::2]
    
    def multiply_fourier_matrices(f1, f2, lA):
        
        import cryspy.tx as tx
        
        if not np.iscomplexobj(f1):
            f1 = f1 + 0j
        if not np.iscomplexobj(f2):
            f1 = f2 + 0j
        
        f = np.zeros(np.size(f1)) + 0j
                
        for l in np.arange(0, lA):
            
            dmi = tx.deg2dim(l)
            dmf = tx.deg2dim(l+1)
            ind = np.arange(dmi, dmf)
            shape = [2 * l + 1, 2 * l + 1]
            f[ind] = np.matrix(np.reshape(f1[ind], shape)) * \
                     np.matrix(np.reshape(f2[ind], shape))

        return f
    
    # -----
    
    cs = component.cs   
    
    c = component.weights / ss.rotations.size / cs.rotations.size
    
    # Ralf original comment:
    #            "for a few center symmetrize before computing c_hat"
    if 10 * component.size * ss.rotations.size * cs.rotations.size < \
                                                   np.amax([kernel.L**3, 100]):
    
        # create a quat object containing all equiv rotations from xtal symm
        g = rot.quat.from_array(rot.consolidate(
                                           component.rotations.symmetrize(cs)))
        
        # add in the sample symmetries
        g = rot.quat.from_array(rot.consolidate(g.symmetrize(ss)))
                                          
        c = np.tile(c, ss.rotations.size * cs.rotations.size)
    
    else:
        g = component.rotations  
    
    # Export Chebyshev coefficients
    endval = np.amin([np.amax([2, kernel.L+1]), np.size(kernel.A)]) - 1
    aa = kernel.A[0:endval]
    
    # calculate Fourier coefficients
    f_hat = gcA2Fourier(g, c, aa)
    
    # for many center symmetrize f_hat
    # XXX: Note that this differs from MTEX 4.1.4 in that the greater than
    #      matches the less than for few centers above.
    if 10 * component.size * ss.rotations.size * cs.rotations.size >= \
                                                   np.amax([kernel.L**3, 100]):
    
        if cs.rotations.size != 1:
            # symmetrize crystal symmetry
            g = cs.rotations
            aa[0:] = 1.0
            c = np.ones(cs.rotations.size)
            f_hat = multiply_fourier_matrices(f_hat, 
                                          gcA2Fourier(g, c, aa), np.size(aa)-1)
            
        if ss.rotations.size != 1:
            # symmetrize specimen symmetry
            g = ss.rotations
            aa[0:] = 1.0
            c = np.ones(ss.rotations.size)
            f_hat = multiply_fourier_matrices(f_hat, 
                                          gcA2Fourier(g, c, aa), np.size(aa)-1)
    
        if component.antipodal == True:
            f_hat = 0.5 * multiply_fourier_matrices(f_hat, 
                                                    f_hat, 
                                                    np.size(aa)-1)

    return f_hat
    
#------------------------------------------------------------------------------    
    
class odf(object):
    
    def __init__(self, orientations, antipodal=False):
        import warnings

        self.rotations = orientations.rotations
        self.size = orientations.size
        self.antipodal = antipodal
        self.weights = np.ones(component.size) # FIXME: This might not be correct for all cases
        
        if len(orientations.cs) == 1:
            self.cs = orientations.cs[0]
        else:
            warnings.warn('Only one phase at a time ' + 
                          'can be analyzed for the ODF')    