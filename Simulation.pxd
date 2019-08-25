cimport Space
cimport Layers
cimport Movement

cdef class Simulation:
    cdef Layers.LiHopfield _olf
    cdef Layers.BAM _asso
    cdef Layers.Single _gus 
    cdef Movement.RadMotor _m
    cdef Space.Space _space
    cdef double[::1] _stim, _I1, _recalled, _ideal_I2, _gus_zeros
    cdef double[:,::1] _pos
    cdef Py_ssize_t _num_orn, _num_grn


cdef class OnlineSim(Simulation):
    cdef object _mapping
    cdef int[::1] _pfunc
    cdef double[::1] _preference, _ideal_preference
    cdef Py_ssize_t _mnpm, _step_counter

    cpdef void set_target(self)

    cpdef void step(self)


cdef class StaticSim(Simulation):
    cdef int _max_num_stim
    cdef int[::1] _pfunc
    cdef double _preference, _ideal_preference
    cdef double[::1] _preference_err
    cdef double[:,::1] _stim_att, _recalling_err

    cpdef void learn(self)

    cpdef void recall(self)