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
        self.act_func = act_func
        self._w = np.random.rand(*shape) if w is None else w
        if not (self.__shape == self._w.shape).all():
            raise ValueError('The shape of the weight matrix must confirm' + \
                             'wtih argument shape.')
        self.out = None

    def get_weight(self):
        return self._w.copy()

    def set_weight(self, w):
        Checking.arg_check(w, 'w', np.ndarray)
        self._w = w

    def shape(self):
        return self.__shape.copy()


class Innate(Layer):
    """ The Innate connections between primary receptor neurons and primary
        interneurons
    """
    def __init__(self, shape, name=None, act_func=expit, w=None):
        """
        Parameters
        ----------
        shape : array-like
            The numbers of receptors and interneurons
        name : str
            The network's name as its identifier. Default is None.
        act_func : callable
            The activation function. Default is signoid.
        w : numpy.ndarray
            The weight matrix. Default is None. If None, w will be randomized.
            w[i, j] is the weight from receptor i to interneuron j
        """
        super().__init__(shape, name, act_func, w)
        if self._w.ndim != 2:
            raise ValueError('The dimension of weight matrix must be 2.')

    def feed(self, inp):
        """ Feed the layer with input

        Parameters
        ----------
        inp: numpy.ndarray
            The sensory input

        Returns
        ----------
        out: numpy.ndarray
            The output
        """
        Checking.arg_check(inp, 'inp', np.ndarray)
        self.out = self.act_func(inp @ self._w)
        return self.out