cimport Space
cimport Layers
cimport Movement

cdef class Simulation:
    cdef Layers.LiHopfield _olf
    cdef Layers.BAM _asso
    cdef Layers.Single _gus 
    cdef Movement.RadMotor _m
    cdef Space.Space _space
    cdef int[::1] _pfunc
    cdef double[::1] _stim, _I1, _recalled, _ideal_I2, _gus_zeros, _preference, _ideal_preference
    cdef double[:,::1] _pos
    cdef object _mapping
    cdef Py_ssize_t _num_orn, _num_grn, _mnpm, _step_counter

    cpdef void set_target(self)

    cpdef void step(self)