#cython: language_level=3
cimport numpy as np
import numpy as np
cimport cython
cimport scipy.linalg.cython_blas as blas
from scipy.special import expit
import matplotlib.pyplot as plt
import networkx as nx


""" flags for BLAS functions 
"""
cdef double NEG1 = -1
cdef double POS1 = 1
cdef double ZERO = 0
cdef char TRANS = b't' # Transpose flag
cdef char NO = b'n' # No/None flag
cdef char LT = b'l' # Left/Lower flag
cdef int LD = 1 # the spacing variable



""" Helper functions
"""
cdef extern from "core.h":
    void tanh_f(double *res, const double *arr, int len, int ax, double th)
    void sig_power(double *res, const double *arr, int max_period, 
                int len, double th)
    void dep_f(double *w, const double *I, const int *shape, double phi)
    void sq_outer(double *res, const double *a1, const double *a2, 
                int len, char tri)
    void I_minus_aL(double *L, double a, int size)
    void rmnan(double *w, int *shape)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _GHA(double[::1,:] W, double[::1,:] L, double[::1,:] O, 
                double eta, int m, int n, bint trans_O=False):
    """ generalized Hebbian alg.
        W += eta * (O - L @ W)

        parameters
        ----------
        W: double, fortran-ordered memoryview
            the weight matrix to update
        L: double, fortran-ordered memoryview
            the lower triangluar part of the outer product of the output
        O: double, fortran-ordered memoryview
            the outer product of the output and the input
        eta: double
            the learning/adapting rate
        m: int
            the number of the rows of W/O and the rows/cols of L
        n: int
            the number of the cols of W/O
        trans_O: bint
            if transposing O
    """

    cdef int mn = m * n

    # L' = I - eta * L
    I_minus_aL(&L[0, 0], eta, m)

    # W = L' @ W
    blas.dtrmm(&LT, &LT, &NO, &NO, &m, &n, &POS1, &L[0, 0], &m, &W[0, 0], &m)

    # W = eta * O + W
    if trans_O:
        blas.daxpy(&mn, &eta, &O.T.copy_fortran()[0, 0], &LD, &W[0, 0], &LD)
    else:
        blas.daxpy(&mn, &eta, &O[0, 0], &LD, &W[0, 0], &LD)



""" Begin class definitions
"""


cdef class Layer:
    """ Super class
    """

    def __init__(self, np.ndarray[np.int_t] shape):
        """
        Parameters
        ----------
        shape : array-like
            The numbers of neurons in each layers
        """

        self._shape = shape

    def shape(self):
        return self._shape.copy()




cdef class Single(Layer):
    """ A single layer
    """

    def __init__(self, int size, object act_func=expit):
        """
        Parameters
        ----------
        size : int
            The numbers of receptors
        act_func : collection of callables
            The activation functions
        """
        self._shape = np.array((size,), dtype=np.int32)
        self._act_func = act_func


    cpdef double[::1] feed(self, double[::1] I):
        """ Feed the layer with input

        Parameters
        ----------
        I: numpy.ndarray
            The sensory input

        Returns
        ----------
        out: numpy.ndarray
            The output
        """
        return self._act_func(I)





cdef class LiHopfield(Layer):
    """ The Li-Hopfield model of olfactory bulb, with the same numbers of mitral
        cells and granule cells
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    def __init__(self, int size, int period=50, int memory=2,
                 double tau=2.0, double adapting_rate=0.0005, 
                 double I_c=0.1, double th=1, bint enable_GHA=False):
        """
        Parameters
        ----------
        size: int
            The numbers of receptors (= mitral cells = granule cells)
        act_func : callable
            The activation function. Default is tanh.
        period: int
            The period during which the agent stay in a spot
        memory: int
            The largest number of period to remember
        tau: float
            The cell time const
        adapting_rate: float
            The rate at which the system gets adapting to the stimulus values
        I_c: float
            The constant central input to the neurons
        th: float
            The firing threshold
        """
        cdef Py_ssize_t i
        cdef np.ndarray[np.float64_t] dim1arr
        cdef np.ndarray[np.float64_t, ndim=2] dim2arr
        # descriptors
        self._size = size
        self._shape = np.array((size, size), dtype=np.int32)
        self._enable_GHA = enable_GHA
        
        # threshold
        self._th = th

        # time related parameters
        self._period = period
        self._tau = tau
        self._eta = adapting_rate 
        # 1 / time const; subtracted by 1 for future convenience
        self._a_x = 1 - 1 / tau
        self._a_y = 1 - 1 / tau

        # central input
        self._I_c = np.full(self._size, I_c, dtype=np.float64).\
                    copy(order='F')

        # buffer arr
        dim1arr = np.zeros(self._size, dtype=np.float64)
        dim2arr = np.zeros(self._shape, dtype=np.float64)


        # input
        self._I = dim1arr.copy(order='F')


        # mitral cells' internal state
        self._x = dim1arr.copy(order='F')
        # mitral cells' external state
        self._Gx = dim1arr.copy(order='F')
        # granule cells' internal state
        self._y = dim1arr.copy(order='F')
        # granule cells' external state
        self._Gy = dim1arr.copy(order='F')
        # mitral cells' signal power over the period
        self._p = dim1arr.copy(order='F')

        # inter-mitral connections
        self._L = dim2arr.copy(order='F')

        
        # helper matrices
        self._xy = dim2arr.copy(order='F')
        self._Lxx = dim2arr.copy(order='F')
        self._Lyy = dim2arr.copy(order='F')

        
        # init L
        for i in range(self._size):
            self._L[i, (i + 1) % self._size] = 1
            self._L[i, (self._size + i - 1) % self._size] = 1
        

        # init GM and MG
        for i in range(self._size):
            dim2arr[i, i] = 1
            dim2arr[i, (i + 1) % self._size] = 1
            dim2arr[i, (self._size + i - 1) % self._size] = 1                

        # connections from mitral to granule cells
        self._GM = dim2arr.copy(order='F')
        # connections from granule to mitral cells
        self._MG = dim2arr.T.copy(order='F')


        # memory-related

        # the largest number of periods to remember
        self._memory = memory

        self._max_period = self._memory * self._period

        # the next interval for the current period
        self._next_intvl = 0

        # for clarifying the increases in input (for naive noise filter)
        self._Gx_record = np.zeros((self._max_period, self._size), 
                                    dtype=np.float64, order='F')

    def set_tau(self, double t):
        self._tau = t
        self._a_x = 1 - 1 / t
        self._a_y = 1 - 1 / t

    def set_adapting_rate(self, double a):
        self._eta = a

    @cython.initializedcheck(False)
    def get_weight(self):
        return self._MG, self._GM, self._L

    @cython.initializedcheck(False)
    def get_Gx_record(self):
        return self._Gx_record


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _update_xy(self):
        """
        x = - MG @ Gy + (1 - a_x) * x + L @ Gx + I
        y =   GM @ Gx + (1 - a_y) * y + I_c

        hard-coded in case of any change in the equations
        """
        cdef int m = self._size

        # calculate x
        # x = - MG @ Gy + (1 - a_x) * x
        blas.dgemv(&NO, &m, &m, &NEG1, &self._MG[0, 0], &m, 
                    &self._Gy[0], &LD, &self._a_x, &self._x[0], &LD)

        # x = x + L @ Gx
        blas.dgemv(&NO, &m, &m, &POS1, &self._L[0, 0], &m, 
                    &self._Gx[0], &LD, &POS1, &self._x[0], &LD)

        # x = x + I
        blas.daxpy(&m, &POS1, &self._I[0], &LD, &self._x[0], &LD)

        # calculate y
        # y =   GM @ Gx + (1 - a_y) * y
        blas.dgemv(&NO, &m, &m, &POS1, &self._GM[0, 0], &m, 
                    &self._Gx[0], &LD, &self._a_y, &self._y[0], &LD)

        # y = y + I_c
        blas.daxpy(&m, &POS1, &self._I_c[0], &LD, &self._y[0], &LD)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef double[::1] feed(self, double[::1] I):
        # 1D array buffer
        cdef np.ndarray[np.float64_t] one_d = \
            np.zeros(self._size, dtype=np.float64)

        # 2D array buffer
        cdef np.ndarray[np.float64_t, ndim=2] two_d = \
            np.zeros(self._shape, dtype=np.float64)

        # iteration variable
        cdef Py_ssize_t t 

        # starting index for _Gx_record
        cdef Py_ssize_t k = self._next_intvl * self._period

        # # init: set the internal states and power to zeros
        # self._x = np.zeros(self._size)
        # self._y = self._x.copy()
        # self._p = np.zeros(self._size)

        self._xy = two_d.copy(order='F')
        self._Lxx = two_d.copy(order='F')
        self._Lyy = two_d.copy(order='F')

        # init: the outputs of mitral and granule cells at time 0
        # copy from Lxx for convenience
        self._Gx = one_d.copy(order='F')
        self._Gy = one_d.copy(order='F')

        # init: the input
        self._I = I


        # start simulation iterations
        for t in range(1, self._period):
            # renew the internal states
            self._update_xy()

            # update the outputs
            tanh_f(&self._Gx[0], &self._x[0], self._size, 0, self._th)
            tanh_f(&self._Gy[0], &self._y[0], self._size, 1, self._th)

            self._Gx_record[k + t - 1] = self._Gx


            # outer products
            # xy = x @ y.T
            sq_outer(&self._xy[0, 0], &self._x[0], 
                    &self._y[0], self._size, NO)
            # Lxx = LT(x @ x.T)
            sq_outer(&self._Lxx[0, 0], &self._x[0], 
                    &self._x[0], self._size, LT)
            # Lyy = LT(y @ y.T)
            sq_outer(&self._Lyy[0, 0], &self._y[0], 
                    &self._y[0], self._size, LT)

            if self._enable_GHA:
                # GHA
                # MG += eta * (xy - Lxx @ MG)
                # GM += eta * (yx - Lyy @ GM)
                _GHA(self._MG, self._Lxx, self._xy, self._eta, 
                        self._size, self._size)
                _GHA(self._GM, self._Lyy, self._xy, self._eta, 
                        self._size, self._size, True)

        # update the power of mitral cells' signals
        sig_power(&self._p[0], &self._Gx_record[0, 0], self._max_period, 
                    self._size, self._th)

        # update the next interval counter
        self._next_intvl += 1
        self._next_intvl %= self._memory

        return self._p


    def save_img(self, rad=0.1, fname='li_hop.png'):
        g = nx.MultiDiGraph()

        # add edges
        for i in range(self._size):
            # Inter-mitral
            g.add_edge(i, (i + 1) % self._size, color='r')
            g.add_edge((i + 1) % self._size, i, color='r')

            for j in (-1, 0, 1):
                j_ = i + self._size + j
                if j_ == self._size - 1:
                    j_ += self._size
                elif j_ == self._size * 2:
                    j_ -= self._size
                # Mitral to granule
                g.add_edge(i, j_, color='r')
                # Granule to mitral
                g.add_edge(j_, i, color='b')

        # Extract a list of edge colors
        edges = g.edges()
        edge_colors = [g[u][v][0]['color'] for u, v in edges]

        # Extract a list of node colors
        node_colors = ['gray' if node < 10 else 'black' for node in g]

        # define fixed positions of nodes
        fixed_pos = {}
        for i in range(20):
            # Mitral cells are on the outer circle ("surface")
            # Granule cells are on the inner circle ("inside")
            r = 1 / (i // self._size + 1) # radius
            x = np.pi / self._size * 2 * (i % self._size) # angle
            fixed_pos.update({i: (r * np.cos(x), r * np.sin(x))})

        # get the nx positions
        pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

        # draw
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 8)
        nx.draw(g, pos=pos, node_color=node_colors, edge_color=edge_colors,
                connectionstyle='arc3,rad={}'.format(rad), ax=ax)
        fig.savefig(fname, dpi=100, bbox_inches='tight', transparent=True)





cdef class BAM(Layer):
    """ Bidirectional associative memory
    """

    def __init__(self, int norn, int ngrn, double adapting_rate=0.001, 
                bint enable_dep=False, double depression_rate=1e-10):
        """
        Parameters
        ----------
        shape : array-like
            The sizes of pattern X and pattern Y
        dep_func : callable
            The synapse depression function.
            Default is f(x, I) = x - phi / (I * v**2 + phi).
        adapting_rate: float
            The rate at which the system gets adapting to the stimulus values
        depression_rate: float
            The rate at which the synapses is decaying due to a lack of activities
        """
        self._eta = adapting_rate
        self._phi = depression_rate

        self._shape = np.array((norn, ngrn), dtype=np.int32)
        
        self._W = np.zeros((ngrn, norn), dtype=np.float64).copy(order='F')

        self._enable_dep = enable_dep

    
    def set_depression_rate(self, double a):
        self._phi = a

    def set_adapting_rate(self, double a):
        self._eta = a

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef void learn(self, double[::1] I1, double[::1] I2):
        cdef np.ndarray[np.float64_t, ndim=2] o = \
            np.zeros(self._shape.T, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] l = \
            np.zeros((self._shape[1], self._shape[1]), dtype=np.float64)
        cdef double[::1,:] L = l.copy(order='F')
        cdef double[::1,:] O = o.copy(order='F')
        
        # for BLAS
        cdef int m = self._shape[1]
        cdef int n = self._shape[0]
        cdef int k = 1

        # O = I2 @ I1.T
        blas.dgemm(&NO, &TRANS, &m, &n, &k, &POS1, &I2[0], 
                    &m, &I1[0], &n, &ZERO, &O[0, 0], &m)
        # L = LT(I2 @ I2.T)
        sq_outer(&L[0, 0], &I2[0], &I2[0], self._shape[1], LT)

        # GHA
        # W += eta * O - L @ W
        _GHA(self._W, L, O, self._eta, self._shape[1], self._shape[0])


        if self._enable_dep:
            # depression
            dep_f(&self._W[0, 0], &I1[0], &self._shape[0], self._phi)

        else: # dep_f already has removing NaN functionality
            rmnan(&self._W[0, 0], &self._shape[0])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef double[::1] recall(self, double[::1] I):
        cdef np.ndarray[np.float64_t] r = \
            np.zeros(self._shape[1], dtype=np.float64)
        cdef double[::1] R = r.copy(order='F')
        cdef int m = self._shape[1]
        cdef int n = self._shape[0]

        # R = W @ I
        blas.dgemv(&NO, &m, &n, &POS1, &self._W[0, 0], &m, 
                    &I[0], &LD, &ZERO, &R[0], &LD)
        # euclidean norm
        cdef double norm = blas.dnrm2(&m, &R[0], &LD)

        if norm != 0:
            norm = 1 / norm
            blas.dscal(&m, &norm, &R[0], &LD)
        return R

    @cython.initializedcheck(False)
    def get_weight(self):
        return self._W


    def save_img(self, fname='bam.png'):
        g = nx.MultiDiGraph()

        # add edges
        fixed_pos = {}
        p = self._shape[0] - self._shape[1] / 2
        for i in range(self._shape[0]):
            for j in range(self._shape[0], self._shape[0] + self._shape[1]):
                g.add_edge(i, j)
                fixed_pos.update({i: (0, i), j: (1, j - p)})

        # get the nx positions
        pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

        # draw
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 8)
        nx.draw(g, pos=pos, node_color='black', edge_color='gray', ax=ax)
        fig.savefig(fname, dpi=100, bbox_inches='tight', transparent=True)
