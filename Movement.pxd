cdef class Motor:
    cdef int _lim
    cdef double _preference, _min_step
    cdef double[::1] _pos

    cpdef double[::1] get_pos(self)
    
    cpdef bint is_at(self, double[::1] target, double th=?)


cdef class RadMotor(Motor):
    """ A motor system using heading radian as its internal representation
        of its direction
    """
    cdef double _h_rad, _target_dir, _prev_preference

    cpdef void set_preference(self, double p)

    cdef void _round_rad(self)

    cpdef double[::1] move(self, int sig_ign=?)

    cpdef void rotate(self, double rad)

    cpdef void heading(self, double[::1] target)
