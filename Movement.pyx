#cython: language_level=3
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt, acos, fmod

cdef double PI = 3.14159265358979
cdef double TWO_PI = 3.14159265358979 * 2

""" Helper functions
"""
cdef extern from "core.h":
    void cmove(double *heading_rad, double *pos, const double preference,
            const double prev_preference, const double minStep, 
            const double lim, const double target_dir, const int sig_ign)


""" Begin class definitions
"""

cdef class Motor:
    """ Super class
    """

    def __init__(self, int lim, double min_step=1, double x=0, double y=0):
        """
        Parameters
        ----------
        lim : int
            The space limit
        min_step: double
            The min step length the BV takes
        x: double
            BV's starting x cordination
        y: double
            BV's starting y cordination
        """

        self._lim = lim
        self._min_step = min_step
        self._pos = np.array((x, y))
        self._preference = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double[::1] get_pos(self):
        return self._pos



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef bint is_at(self, double[::1] target, double th=0):
        """
        To check if the BV is at the target position

        Parameters
        ----------
        target: np.ndarray or Memoryview
            The target to check
        th: double
            The threshold for determining if the BV is at the target position
        """

        dist = sqrt((self._pos[0] - target[0])**2 + \
                    (self._pos[1] - target[1])**2)
        return dist <= th


cdef class RadMotor(Motor):
    """ A motor system using heading radian as its internal representation
        of its direction
    """

    def __init__(self, int lim, double h_rad=0, double min_step=1, double x=0, 
                        double y=0):
        """
        Parameters
        ----------
        lim: int
            The space limit
        h_rad: float
            The heading direction, in radian
        min_step: double
            The min step length the BV takes
        x: float
            BV's starting x cordination
        y: float
            BV's starting y cordination
        """

        super().__init__(lim, min_step, x, y)
        
        # the heading angle in radian
        self._h_rad = h_rad
        
        # the previous preference
        self._prev_preference = 0

        # the direction of the target, in radian
        self._target_dir = 0


    def get_heading_rad(self):
        return self._h_rad


    cpdef void set_preference(self, double p):
        """
        Parameters
        ----------
        p: double
            Preference
        """

        self._prev_preference = self._preference
        self._preference = p


    cdef void _round_rad(self):
        if self._h_rad > TWO_PI:
            self._h_rad = fmod(self._h_rad, TWO_PI);
        elif (self._h_rad < 0):
            self._h_rad = fmod(self._h_rad, TWO_PI);
            self._h_rad += TWO_PI;


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double[::1] move(self, int sig_ign=0):
        """
        To move based on (or not based on due to sig_ign) preference

        Parameters
        ----------
        sig_ign: int
            Ignorance signal
            If 1, ignore any moving instruction and maintain 
            its previous heading radian; if 0, follow the moving 
            instructions normally.
        """

        cmove(&self._h_rad, &self._pos[0], self._preference, 
            self._prev_preference, self._min_step, 
            <double> self._lim, self._target_dir, sig_ign)

        return self._pos


    cpdef void rotate(self, double rad):
        self._h_rad += rad 
        self._round_rad()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef void heading(self, double[::1] target):
        """
        Heading towards a target position
        
        Parameters
        ----------
        target: np.ndarray or memoryview
            The target position
        """
        cdef double xdiff = target[0] - self._pos[0]
        cdef double ydiff = target[1] - self._pos[1] 
        if ydiff > 0:
            self._h_rad = acos(xdiff / sqrt(xdiff**2 + ydiff**2))
        else:
            self._h_rad = acos(-xdiff / sqrt(xdiff**2 + ydiff**2)) + PI
        self._target_dir = self._h_rad
