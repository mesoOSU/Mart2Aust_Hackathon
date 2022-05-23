from cffi import FFI
import numpy as np
"""
Not fully implemented yet, has the skeleton of what would be needed 
to get the c functions inside of ODF.c to work as a python pacakge. 
"""



ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""
    mutable struct nfft_plan;
    void nfft_init_1d(nfft_plan myplan,int N,int M);

""")

# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source("TesterNFFT",
                      """
                      
                           #include "nfft3.h"   // the C header of the library
                      
                      """,
                      libraries=['libnfft3'])  # library name, for the linker


def to_python(arr):
    buffer_size = arr.size * np.complex.__itemsize__
    pass


def to_cpp(arr):
    if isinstance(arr, (np.ndarray, np.generic)):
        return ffibuilder.cast('double*', arr.ctypes.data)
    else:
        return ffibuilder.cast('double*', np.array(arr).ctypes.data)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)