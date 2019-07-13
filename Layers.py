#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION: 0.1
#PYTHON_VERSION: 3.6
'''
DESCRIPTION
    A collection of different types of layer
'''
import numpy as np
from scipy.special import expit
import Checking
import matplotlib.pyplot as plt
import networkx as nx

class Layer:
    """ Super class
    """
    def __init__(self, shape, w, act_func, name=None):
        """
        Parameters
        ----------
        shape : array-like
            The numbers of neurons in each layers
        w : numpy.ndarray
            The weight matrix.
        act_func : callable
            The activation function. Default is signoid.
        name : str
            The network's name as its identifier. Default is None.
        """
        #================== Argument Check ============================
        Checking.arg_check(shape, 'shape', (list, np.ndarray))
        Checking.arg_check(name, 'name', str)
        Checking.arg_check(act_func, 'act_func', 'callable')
        Checking.arg_check(w, 'w', np.ndarray)
        #==============================================================

        self._shape = np.array(shape, dtype=int)
        self.name = name
        self._act_func = act_func
        self._w = w
        if not (self._shape == self._w.shape).all():
            raise ValueError('The shape of the weight matrix must confirm' + \
                             'wtih argument shape.')

    def get_weight(self):
        """ Return a copy of the weight matrix. The copy could be a shallow copy.
        """
        if self._w is None:
            raise RuntimeError('Discarded method "get_weight()"')
        return self._w.copy()

    def set_weight(self, w):
        if self._w is None:
            raise RuntimeError('Discarded method "get_weight()"')
        Checking.arg_check(w, 'w', np.ndarray)
        self._w = w

    def shape(self):
        return self._shape.copy()


class Single(Layer):
    """ A single layer
    """
    def __init__(self, size, act_func=expit, name=None):
        """
        Parameters
        ----------
        size : int
            The numbers of receptors
        act_func : collection of callables
            The activation functions
        name : str
            The network's name as its identifier. Default is None.
        """
        #================== Argument Check ============================
        Checking.arg_check(act_func, 'act_func', 'callable')
        Checking.arg_check(name, 'name', str)
        #==============================================================
        self._shape = np.array((size,), dtype=int)
        self.name = name
        self._act_func = act_func
        self._w = None


    def feed(self, I):
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
        Checking.arg_check(I, 'I', np.ndarray)
        self.out = self._act_func(I)
        return self.out


class LiHopfield(Layer):
    """ The Li-Hopfield model of olfactory bulb, with the same numbers of mitral
        cells and granule cells
    """
    def __init__(self, size, name=None, act_func=None, period=50, tau=2,
                 adapting_rate=0.0005, I_c=0.1, th=1):
        """
        Parameters
        ----------
        size: int
            The numbers of receptors (= mitral cells = granule cells)
        name : str
            The network's name as its identifier. Default is None.
        act_func : callable
            The activation function. Default is tanh.
        period: int
            The period during which the agent stay in a spot
        tau: float
            The cell time const
        adapting_rate: float
            The rate at which the system gets adapting to the stimulus values
        I_c: float
            The constant central input to the neurons
        th: float
            The firing threshold
        """
        #================== Argument Check ============================
        Checking.arg_check(name, 'name', str)

        if act_func is None:
            self._act_func = self.tanh_func()
        else:
            self._act_func = act_func
            Checking.collection(act_func, 'act_func', 'callable', 2)
        #==============================================================

        # descriptors
        self._size = int(size)
        self._shape = np.array((size, size), dtype=int)
        self.name = name


        self._w = None
        self._th = float(th)

        # time related parameters
        self._period = int(period)
        self._tau = float(tau)
        self._eta = float(adapting_rate)
        # 1 / time const
        self._a_x = 1 / tau
        self._a_y = 1 / tau

        self._I_c = float(I_c) # central input

        # inter-mitral connections
        self._L = np.zeros((self._size, self._size))
        for i in range(self._size):
            self._L[i, (i + 1) % self._size] = 1
            self._L[i, i - 1] = 1

        # mitral cells' internal state
        self._x = np.zeros(self._size)
        # granule cells' internal state
        self._y = self._x.copy()
        # mitral cells' signal power over the period
        self._p = self._x.copy()

        # connections from granule to mitral cells
        self._GM = np.zeros(self._shape)
        for i in range(self._size):
                indices = np.arange(i - 1, i + 2)
                larger = np.argwhere(indices >= self._size)
                indices[larger] = indices[larger] % self._size
                self._GM[i, indices] = 1

        # connections from mitral to granule cells
        self._MG = self._GM.T.copy()

    def get_weight(self):
        return self._MG.copy(), self._GM.copy(), self._L.copy()


    def tanh_func(self):
        # activation function parameters
        sx, sx_ = 1.4, 0.14
        sy, sy_ = 2.9, 0.29

        # helper step function
        P = lambda a: np.piecewise(a, [a < self._th,
                                       a >= self._th], [0.1, 1])

        # activation functions; _sub means sub-threshold
        G_x = lambda a: sx_ + (sx * np.tanh((a - self._th) / sx / P(a))) * P(a)
        G_y = lambda a: sy_ + (sy * np.tanh((a - self._th) / sy / P(a))) * P(a)

        return (G_x, G_y)

    def feed(self, I):
#        # init: set the internal states and power to zeros
#        self._x = np.zeros(self._size)
#        self._y = self._x.copy()
#        self._p = self._x.copy()

        # init: the outputs of mitral and granule cells at time 0
        G_x = np.zeros(self._size)
        G_y = G_x.copy()

        # start simulation iterations
        for t in range(1, self._period):
            # renew the internal states
            self._x += - G_y @ self._GM \
                        - self._a_x * self._x + G_x @ self._L + I
            self._y += G_x @ self._MG - self._a_y * self._y + self._I_c

            # update the outputs
            G_x = self._act_func[0](self._x)
            G_y = self._act_func[1](self._y)

            # update the power of mitral cells' signals
            p = np.piecewise(G_x, [G_x < self._th, G_x >= self._th], [0, 0.5])
            self._p += (G_x * p)**2

            # outer products
            yx = np.outer(self._y, self._x)
            Lxx, Lyy = [np.triu(np.outer(a, a)) for a in (self._x, self._y)]

            # GHA
            self._GM += self._eta * (yx - self._GM @ Lxx)
            self._MG += self._eta * (yx.T - self._MG @ Lyy)

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


class BAM(Layer):
    """ Bidirectional associative memory
    """
    def __init__(self, shape, name=None, act_func=lambda x: x, dep_func=None,
                 adapting_rate=0.001, depression_rate=1e-10):
        """
        Parameters
        ----------
        shape : array-like
            The sizes of pattern X and pattern Y
        name : str
            The network's name as its identifier. Default is None.
        act_func : callable
            The activation function. Default is f(x) = x.
        dep_func : callable
            The synapse depression function.
            Default is f(x, I) = x - phi / (I * v**2 + phi).
        adapting_rate: float
            The rate at which the system gets adapting to the stimulus values
        depression_rate: float
            The rate at which the synapses is decaying due to a lack of activities
        """
        self._eta = float(adapting_rate)
        self._phi = float(depression_rate)

        self._act_func = act_func

        if dep_func is None:
            self._dep_func = lambda v, I: v - self._phi / (I * v**2 + self._phi)
        else:
            self._dep_func = dep_func

        self._shape = np.array(shape)
        self.name = name

        self._w = np.zeros(shape)

    def learn(self, I1, I2):
        # GHA
        self._w += self._eta * (np.outer(I1, I2) \
                    - self._w @ np.triu(np.outer(I2, I2)))
        # depression
        self._w = np.apply_along_axis(lambda v: self._dep_func(v, I1), 0, self._w)

    def recall(self, I):
        r = self._act_func(I) @ self._w
        norm_r = np.linalg.norm(r)
        if norm_r != 0:
            return r / norm_r
        else:
            return r

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
