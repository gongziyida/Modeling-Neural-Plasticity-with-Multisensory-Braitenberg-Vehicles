""" flags for BLAS functions 
"""
cdef double NEG1 = -1
cdef double POS1 = 1
cdef double ZERO = 0
cdef char TRANS = b't' # Transpose flag
cdef char NO = b'n' # No/None flag
cdef char LT = b'l' # Left/Lower flag
cdef int LD = 1 # the spacing variable


cdef void _GHA(double[::1,:] W, double[::1,:] L, double[::1,:] O, 
                double eta, int m, int n, bint trans_O=?)



""" Begin class definitions
"""


cdef class Layer:
    """ Super class
    """
    cdef int[::1] _shape


cdef class Single(Layer):
    """ A single layer
    """
    cdef object _act_func

    cpdef double[::1] feed(self, double[::1] I)




cdef class LiHopfield(Layer):
    """ The Li-Hopfield model of olfactory bulb, with the same numbers of mitral
        cells and granule cells
    """
    cdef bint _enable_GHA
    cdef int _size, _period, _memory, _max_period, _next_intvl
    cdef double _th, _tau, _eta, _a_x, _a_y
    cdef double[::1] _x, _y, _p, _Gx, _Gy, _I, _I_c
    cdef double[::1,:] _L, _GM, _MG, _Lxx, _Lyy, _xy, _Gx_record

    cdef void _update_xy(self)

    cpdef double[::1] feed(self, double[::1] I)




cdef class BAM(Layer):
    """ Bidirectional associative memory
    """
    cdef bint _enable_dep
    cdef double _eta, _phi
    cdef double[::1,:] _W

    cpdef void learn(self, double[::1] I1, double[::1] I2)

    cpdef double[::1] recall(self, double[::1] I)