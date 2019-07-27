#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, acos, fmod

cdef double PI = 3.14159265358979
cdef double TWO_PI = 3.14159265358979 * 2

""" Helper functions
"""
cdef extern from "core.h":
    void cmove(double *h_rad, double *pos, const double p, 
            const double minStep, const double lim, 
            const double target_dir)


""" Begin class definitions
"""

cdef class Motor:
    cdef int _lim
    cdef double _preference, _minStep
    cdef double[::1] _pos

    def __init__(self, int lim, double minStep=1, double x=0, double y=0):
        self._lim = lim
        self._minStep = minStep
        self._pos = np.array((x, y))
        self._preference = 0

    cpdef double[::1] get_pos(self):
        return self._pos

    cpdef bint is_at(self, int[::1] target, double th=0):
        dist = sqrt((self._pos[0] - target[0])**2 + \
                    (self._pos[1] - target[1])**2)
        return dist <= th


cdef class RadMotor(Motor):
    cdef double _h_rad, _prev_preference
    """ A motor system using heading radian as its internal representation
        of its direction
    """
    cdef double _target_dir

    def __init__(self, int lim, double h_rad=0, int minStep=1, 
                    double x=0, double y=0):
        super().__init__(lim, minStep, x, y)
        # the heading angle in radian
        self._h_rad = h_rad
        # the previous preference
        self._prev_preference = 0
        # the direction of the target, in radian
        self._target_dir = 0


    def get_heading_rad(self):
        return self._h_rad


    cpdef void set_preference(self, double p):
        self._prev_preference = self._preference
        self._preference = p


    cdef void _round_rad(self):
        if self._h_rad > TWO_PI:
            self._h_rad = fmod(self._h_rad, TWO_PI);
        elif (self._h_rad < 0):
            self._h_rad = fmod(self._h_rad, TWO_PI);
            self._h_rad += TWO_PI;


    cpdef void move(self):
        cdef double p = self._preference - self._prev_preference
        cmove(&self._h_rad, &self._pos[0], p, self._minStep, 
            <double> self._lim, self._target_dir)


    cpdef void rotate(self, double rad):
        self._h_rad += rad 
        self._round_rad()


    cpdef void heading(self, double targetx, double targety):
        cdef double xdiff = targetx - self._pos[0]
        cdef double ydiff = targety - self._pos[1] 
        if ydiff > 0:
            self._h_rad = acos(xdiff / sqrt(xdiff**2 + ydiff**2))
        else:
            self._h_rad = acos(-xdiff / sqrt(xdiff**2 + ydiff**2)) + PI
        self._target_dir = self._h_rad
