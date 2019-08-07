""" Comparison functions as the parameters for C function qsort
"""
# compare by x
cdef int cmp_ax0(const void* a, const void* b) nogil

# compare by y
cdef int cmp_ax1(const void* a, const void* b) nogil


""" Begin class definitions
"""

cdef class Space:
    cdef int _num_stim, _num_orn, _num_grn, _max_pos, _gus_T, _pixel_dim
    cdef LazyKDTree _kd
    cdef str _method
    cdef int[:,::1] _pos
    cdef double[:,::1] _att
    cdef double[:,:,::1] _space

    cpdef double[::1] stim_at(self, Py_ssize_t x, Py_ssize_t y)




""" Space partitioning classes
"""

cdef class Node:
    """ A helper class for KD tree
    """
    cdef public int[::1] pos
    cdef public Node lc, rc
    cdef public bint flag # if visited



cdef class LazyKDTree:
    """ A non-deterministic KD tree that does not give exactly 
        the nearest neighbor so as to save some runtime
    """
    cdef int _num_stim, _max_pos, _num_visited
    cdef bint _flag # buffer flag
    cdef Node _tree

    cdef Node _build(self, int[:,::1] pos, Py_ssize_t start, 
                     Py_ssize_t end, int ax)

    cpdef double[::1] near(self, double[::1] pos)

    cdef void _near(self, double[::1] pos, Node cur, 
                    list local_min, Py_ssize_t dim)

    cdef double _dist_to(self, Node n, double[::1] pos)