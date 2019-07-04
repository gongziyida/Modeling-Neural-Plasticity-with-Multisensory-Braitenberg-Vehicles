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
    def __init__(self, shape, name=None, act_func=expit, w=None):
        """
        Parameters
        ----------
        shape : array-like
            The numbers of neurons in each layers
        name : str
            The network's name as its identifier. Default is None.
        act_func : callable
            The activation function. Default is signoid.
        w : numpy.ndarray
            The weight matrix. Default is None. If None, w will be randomized.
        """
        #================== Argument Check ============================
        Checking.arg_check(shape, 'shape', (list, np.ndarray))
        Checking.arg_check(name, 'name', str)
        Checking.arg_check(act_func, 'act_func', 'callable')
        Checking.arg_check(w, 'w', np.ndarray)
        #==============================================================

        self.__shape = np.array(shape)
        self.name = name
        self.__act_func = act_func
        self._w = np.random.rand(*shape) if w is None else w
        if not (self.__shape == self._w.shape).all():
            raise ValueError('The shape of the weight matrix must confirm' + \
                             'wtih argument shape.')
        self.out = None

    def get_weight(self):
        """ Return a copy of the weight matrix. The copy could be a shallow copy.
        """
        if self._w is None:
            raise RuntimeError('Discarded method "get_weight()"')
        return self._w.copy()

    def set_weight(self, w):
        Checking.arg_check(w, 'w', np.ndarray)
        self._w = w

    def shape(self):
        return self.__shape.copy()


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
        self.__shape = np.array([size, 1], dtype=int)
        self.name = name
        self.__act_func = act_func
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
        self.out = self.__act_func(I)
        return self.out


class LiHopfield(Layer):
    """ The Li-Hopfield model of olfactory bulb, with the same numbers of mitral
        cells and granule cells
    """
    def __init__(self, size, name=None, act_func=None, period=50, tau=2,
                 adapting_rate=0.00001, I_c=0.1, th=1):
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
            self.__act_func = self.tanh_func()
        else:
            self.__act_func = act_func
            Checking.collection(act_func, 'act_func', 'callable', 2)
        #==============================================================

        # descriptors
        self.__size = int(size)
        self.__shape = [size, size]
        self.name = name


        self._w = None
        self.__th = float(th)

        # time related parameters
        self.__period = int(period)
        self.__tau = float(tau)
        self.__eta = float(adapting_rate)
        # 1 / time const
        self.__a_x = 1 / tau
        self.__a_y = 1 / tau

        self.__I_c = float(I_c) # central input

        # inter-mitral connections
        self.__L = np.zeros((self.__size, self.__size))
        for i in range(self.__size):
            self.__L[i, (i + 1) % self.__size] = 1
            self.__L[i, i - 1] = 1

        # mitral cells' internal state
        self.__x = np.zeros(self.__size)
        # granule cells' internal state
        self.__y = self.__x.copy()
        # mitral cells' signal power over the period
        self.__p = self.__x.copy()

        # connections from granule to mitral cells
        self.__GM = np.zeros(self.__shape)
        for i in range(self.__size):
                indices = np.arange(i - 1, i + 2)
                larger = np.argwhere(indices >= self.__size)
                indices[larger] = indices[larger] % self.__size
                self.__GM[i, indices] = 1

        # connections from mitral to granule cells
        self.__MG = self.__GM.T.copy()


    def tanh_func(self):
        # activation function parameters
        sx, sx_ = 1.4, 0.14
        sy, sy_ = 2.9, 0.29

        # helper step function
        P = lambda a: np.piecewise(a, [a < self.__th,
                                       a >= self.__th], [0.1, 1])

        # activation functions; _sub means sub-threshold
        G_x = lambda a: sx_ + (sx * np.tanh((a - self.__th) / sx / P(a))) * P(a)
        G_y = lambda a: sy_ + (sy * np.tanh((a - self.__th) / sy / P(a))) * P(a)

        return (G_x, G_y)

    def feed(self, I):
#        # init: set the internal states and power to zeros
#        self.__x = np.zeros(self.__size)
#        self.__y = self.__x.copy()
#        self.__p = self.__x.copy()

        # init: the outputs of mitral and granule cells at time 0
        G_x = np.zeros(self.__size)
        G_y = G_x.copy()

        # start simulation iterations
        for t in range(1, self.__period):
            # renew the internal states
            self.__x += - G_y @ self.__GM \
                        - self.__a_x * self.__x + G_x @ self.__L + I
            self.__y += G_x @ self.__MG - self.__a_y * self.__y + self.__I_c

            # update the outputs
            G_x = self.__act_func[0](self.__x)
            G_y = self.__act_func[1](self.__y)

            # update the power of mitral cells' signals
            p = np.piecewise(G_x, [G_x < self.__th, G_x >= self.__th], [0, 0.5])
            self.__p += (G_x * p)**2

            # outer products
            yx = np.outer(self.__y, self.__x)
            Lxx, Lyy = [np.triu(np.outer(a, a)) for a in (self.__x, self.__y)]

            # GHA
            self.__GM += self.__eta * (yx - self.__GM @ Lxx)
            self.__MG += self.__eta * (yx.T - self.__MG @ Lyy)

        return self.__p

    def save_img(self, rad=0.1, fname='li_hop.png'):
        g = nx.MultiDiGraph()

        # add edges
        for i in range(self.__size):
            # Inter-mitral
            g.add_edge(i, (i + 1) % self.__size, color='r')
            g.add_edge((i + 1) % self.__size, i, color='r')

            for j in (-1, 0, 1):
                j_ = i + self.__size + j
                if j_ == self.__size - 1:
                    j_ += self.__size
                elif j_ == self.__size * 2:
                    j_ -= self.__size
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
            r = 1 / (i // self.__size + 1) # radius
            x = np.pi / self.__size * 2 * (i % self.__size) # angle
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
                 adapting_rate=0.001, depression_rate=4e-6):
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
            The synapse depression function. Default is f(x, I) = x - phi / (I + 0.001).
        adapting_rate: float
            The rate at which the system gets adapting to the stimulus values
        depression_rate: float
            The rate at which the synapses is decaying due to a lack of activities
        """
        self.__eta = float(adapting_rate)
        self.__phi = float(depression_rate)

        if dep_func is None:
            self.__dep_func = lambda v, I: v - self.__phi / (I + 0.001)
        else:
            self.__dep_func = dep_func

        super().__init__(shape, name, act_func, np.zeros(shape))
        if self._w.ndim != 2:
            raise ValueError('The dimension of weight matrix must be 2.')

    def learn(self, I1, I2):
        # GHA
        self._w += self.__eta * (np.outer(I1, I2) \
                    - self._w @ np.triu(np.outer(I2, I2)))
        # depression
        self._w = np.apply_along_axis(lambda v: self.__dep_func(v, I1), 0, self._w)

    def recall(self, I):
        r = I @ self._w
        return (r + 0.001) / (np.linalg.norm(r) + 0.001)