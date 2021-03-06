#cython: language_level=3
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport abs, sqrt, ceil, exp as cabs, sqrt, ceil, exp
from libc.stdlib cimport qsort
import matplotlib.pyplot as plt


""" Helper functions for building cores 
"""

cdef extern from "core.h":
    void set_stim_pos(int *pos, int num_stim, int max_pos, char method)
    void build_space(double *space, const double *att, const int *pos, 
                    int num_stim, int max_pos, int num_orn, 
                    int pxl_dim, int gus_T)


""" Comparison functions as the parameters for C function qsort
"""
# compare by x
cdef int cmp_ax0(const void* a, const void* b) nogil:
    cdef int* p1 = <int*> a
    cdef int* p2 = <int*> b
    return p1[0] - p2[0]

# compare by y
cdef int cmp_ax1(const void* a, const void* b) nogil:
    cdef int* p1 = <int*> a
    cdef int* p2 = <int*> b
    return p1[1] - p2[1]


""" Begin class definitions
"""

cdef class Space:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __init__(self, int num_stim, int num_orn, int num_grn, 
                 object mapping, str method='r',
                 int max_pos=200, int gus_T=5):
        """
        Parameters
        ----------
        num_stim: int
            The number of stimuli to be initialized
        end_orn: int
            The number of olfactory attributes in a stimulus
        num_grn: int
            The number of gustatory attributes in a stimulus
        mapping: collection of callables
            Mapping from olfactory attributes to gustatory attributes
            It should be of form "mapping(olf, gus)"
        method: str
            The method to choose the locations of stimulus sources. Default is
            'random'. Another option can be 'matrix'.
        max_pos: int
            The maximum value in each axis
        gus_T: int
            The threshold within which the gustatory information is detectable
        """

        assert num_stim >= 1
        self._max_pos = max_pos
        self._gus_T = gus_T
        self._num_stim = num_stim
        self._num_orn = num_orn
        self._num_grn = num_grn
        self._pixel_dim = num_orn + num_grn
        self._mapping = mapping

        # the positions of the stimulus sources based on the method specified
        self._pos = np.zeros((num_stim, 2), dtype=np.int32) 

        # the attributes of the stimulus sources        
        self._att = np.zeros((num_stim, self._pixel_dim), dtype=np.float64)

        # the static space
        self._space = np.zeros((self._max_pos, self._max_pos, self._pixel_dim),
                                dtype=np.float64)
        
        self.mk_space(method)


    def size(self):
        return self._num_stim

    def get_num_receptors(self):
        return self._num_orn, self._num_grn

    def get_stim_pos(self):
        return self._pos

    def get_stim_att(self):
        return self._att

    def get_max_pos(self):
        return self._max_pos

    def mk_space(self, method='r'):
        """
        To make a space based on the method given 
        
        Parameters
        ----------
        method: str
            The method to choose the locations of stimulus sources. Default is
            'random'. Another option can be 'matrix'.
        """

        cdef char method_char = b'r' if method == 'r' else b'm'
        cdef double[:,::1] olf_att, gus_att
        cdef Py_ssize_t i

        # the positions of the stimulus sources based on the method specified
        set_stim_pos(&self._pos[0, 0], self._num_stim, self._max_pos, method_char)

        # the kd tree for searching the stimuli
        self._kd = LazyKDTree(self._pos, self._max_pos)

        # olfactory attributes
        olf_att = np.abs(np.random.normal(size=(self._num_stim, self._num_orn)))
        # gustatory attributes
        gus_att = np.zeros((self._num_stim, self._num_grn))
        
        # mapping(olf_att, gus_att)
        for i in range(olf_att.shape[0]):
            self._mapping(olf_att[i], gus_att[i])

        # assign back
        self._att[:, :self._num_orn] = olf_att
        self._att[:, self._num_orn:] = gus_att

        # build static stimulus environment
        build_space(&self._space[0, 0, 0], &self._att[0, 0], &self._pos[0, 0], 
                    self._num_stim, self._max_pos, self._num_orn, 
                    self._pixel_dim, self._gus_T)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef double[::1] stim_at(self, Py_ssize_t x, Py_ssize_t y):
        # round pos
        x %= self._max_pos
        y %= self._max_pos

        return self._space[x, y]


    def near(self, pos):
        return self._kd.near(pos)


    def save_img(self, name_prefices=('odor_space', 'taste_space')):
        cdef Py_ssize_t i, init, end, k

        for i in (0, 1):
            # the largest pixel value
            if i == 0:
                vmax = np.max(self._space[:, :, :self._num_orn])
                init, end = 0, self._num_orn
            else:
                vmax = np.max(self._space[:, :, self._num_orn:])
                init, end = self._num_orn, self._num_grn

            for k in range(end):
                plt.clf()
                plt.rcParams["figure.figsize"] = [8, 8]

                fname = name_prefices[i] + '_{}.png'.format(k) # file name

                # heatmap
                im = plt.imshow(np.asarray(self._space[:, :, init + k]).T,
                                vmin=0, vmax=vmax, origin='lower')
                plt.colorbar(im, fraction=0.02, pad=0.01)

                if init == 0:
                    # stimulus source locations
                    plt.scatter(*self._pos.T, s=5, c='r')

                plt.axis('off') # no need for axes
                plt.savefig(fname, dpi=100, bbox_inches='tight',
                            transparent=True)


""" Space partitioning classes
"""

cdef class Node:
    """ A helper class for KD tree
    """

    def __init__(self, int[::1] pos, Node lc, Node rc):
        self.pos = pos
        self.lc = lc
        self.rc = rc
        self.flag = False



cdef class LazyKDTree:
    """ A non-deterministic KD tree that does not give exactly 
        the nearest neighbor so as to save some runtime
    """

    def __init__(self, int[:,::1] pos, int max_pos):
        """
        Parameters
        ----------
        pos: numpy.ndarray
            The positions of the stimuli to feed in this KD Tree
        max_pos: int
            The maximum value in each axis
        """

        self._num_stim = pos.shape[0]
        self._max_pos = max_pos


        self._tree = self._build(pos, 0, self._num_stim, 0)

        self._num_visited = 0
        self._flag = True


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Node _build(self, int[:,::1] pos, Py_ssize_t start, 
                     Py_ssize_t end, int ax):
        cdef int rang = end - start # range
        cdef Py_ssize_t ax_new = 1 - ax # next axis
        cdef Py_ssize_t mid = start + rang // 2
        cdef int[::1] val
        cdef Node lc, rc

        if rang == 1: # one leaf
            return Node(pos[start], None, None)
        elif rang == 0: # no leaf
            return None
        
        # sort the entries based on axis <ax>
        if ax == 0:
            qsort(&pos[start, 0], rang, pos.strides[0], &cmp_ax0)
        else:
            qsort(&pos[start, 0], rang, pos.strides[0], &cmp_ax1)

        # value of the branching node to be constructed
        val = pos[mid]

        # go to the left and right children
        lc = self._build(pos, start, mid, ax_new)
        rc = self._build(pos, mid+1, end, ax_new)

        return Node(val, lc, rc)


    cpdef double[::1] near(self, double[::1] pos):
        """ Find a nearby point

        Parameters
        ----------
        pos: array-like
            The position of the query point
        _eap: boolean
            Enable printing details

        Returns
        ----------
        pos_nearby: array-like
            The position of a nearby point
        """
        cdef Node cur = self._tree
        # TODO: change the list to a pointer (memoryview?)
        # note down the local minimum
        cdef list local_min = [cur, self._dist_to(cur, pos)] 

        self._near(pos, cur, local_min, 0)

        # designed for BV: not to "obsess" on one source
        self._num_visited += 1
        if self._num_visited == self._num_stim:
            self._num_visited = 0
            self._flag = not self._flag

        cur = local_min[0]
        cur.flag = self._flag

        cdef double[::1] res = np.array(cur.pos, dtype=np.float64)
        return res


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _near(self, double[::1] pos, Node cur, 
                    list local_min, Py_ssize_t dim):
        cdef double to_cur = self._dist_to(cur, pos)
        cdef double to_line
        cdef Node first, second

        ## _eap: boolean
        ##       reserved for testing

        # if _eap: # enable printing traversal processes
        #     print('current: ', cur.pos)
        #     print('lc: ', 'X' if cur.lc is None else cur.lc.pos)
        #     print('rc: ', 'X' if cur.rc is None else cur.rc.pos)
        #     print('current local min: ', local_min[1])
        #     print('---')

        if pos[dim] < cur.pos[dim]:
            first, second = cur.lc, cur.rc
        else:
            second, first = cur.lc, cur.rc

        if not (first is None):
            self._near(pos, first, local_min, 1 - dim)
        # reaching the end; about to backtrack
        elif cur.flag != self._flag:
            local_min[0], local_min[1] = cur, to_cur

        # backtracking
        if to_cur < local_min[1] and  cur.flag != self._flag:
            local_min[0], local_min[1] = cur, to_cur

        if not (second is None):
            to_line = cabs(pos[dim] - second.pos[dim])
            if to_line < local_min[1]:
                self._near(pos, second, local_min, 1 - dim)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _dist_to(self, Node n, double[::1] pos):
        cdef int x = <int> cabs(pos[0] - n.pos[0])
        cdef int y = <int> cabs(pos[1] - n.pos[1])

        # when the agent hits a boundary, it continues on the other side.
        # Therefore, the differences in coordinates from both directions are
        # considered.
        if x > self._max_pos / 2:
            x = self._max_pos - x

        if y > self._max_pos / 2:
            y = self._max_pos - y
        
        return sqrt(x**2 + y**2)