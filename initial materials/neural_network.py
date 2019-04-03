#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#DATE: Wed Apr  3 16:12:16 2019
#VERSION: 0.1
#PYTHON_VERSION: 3.6
'''
DESCRIPTION
A simple testing design of neural network.
'''
import numpy as np

class neural_network:
    def __init__(self, connections=None, ORN_size=6, GRN_size=3,
                 pos=(0, 0), stimuli=None, threshold=0.1):
        """Constructing a single neural network

        Parameters
        ----------
        connection : numpy.ndarray
            A connection matrix for all neurons. The top left of the matrix
            must be receptors (ORNs followed by GRNs).
        ORN_size : int
            The size of olfactory receptor neurons.
        GRN_size : int
            The size of gustatory receptor neurons.
        pos : tuple, list, or numpy.ndarray
            The position of the agent in the 2-dimensional space
        stimuli : numpy.ndarray
            A collection of stimuli. It should be a 2-dimensional array,
            with each row representing a stimulus source. The first columns
            must be odor attributes, followed by taste attributes, and then
            x and y values. The number of stimulus attributes must match receptor sizes.
        """
        #================== Argument Check ============================
        if connections is not None and not isinstance(connections, np.ndarray):
            raise ValueError('The connection matrix must be a numpy array.')
        if stimuli is not None and not isinstance(stimuli, np.ndarray):
            raise ValueError('The stimulus matrix must be a numpy array.')
        if not all(isinstance(i, int) for i in (ORN_size, GRN_size)):
            raise ValueError('ORN size and GRN size must be integers.')
        if not isinstance(pos, (tuple, list, np.ndarray)):
            raise ValueError('pos must be a tuple, list, or nparray.')
        if connections is not None: # if the matrix is not None
            if connections.shape[0] < ORN_size + GRN_size:
                raise ValueError('Connection matrix must be at least {} by {}'. \
                                 format(ORN_size + GRN_size))
        #===============================================================

        self.connections = connections if connections is not None \
                                        else np.zeros(ORN_size + GRN_size)
        self.ORN_size = ORN_size
        self.GRN_size = GRN_size
        self.pos = pos
        self.stimuli = stimuli
        self.threshold = threshold
        self.sensory_inputs = self._sensory_inputs()

    def _sensory_inputs(self):
        """Calculating sensory inputs
        """
        if self.stimuli is None: return # stimuli == None

        sensory_inputs = np.copy(self.stimuli)
        for s in sensory_inputs:
            dist = np.linalg.norm(self.pos - s[-2 :])
            s[: self.ORN_size] /= dist

            heaviside = 1 if dist <= self.threshold else 0
            s[self.ORN_size: self.ORN_size + self.GRN_size] *= heaviside

        return np.sum(sensory_inputs[:, :-2], axis=0)

    def get_sensory_inputs(self):
        return self.sensory_inputs

    def jump(self, pos):
        """Jump to position pos

        Parameters
        ----------
        pos : tuple, list, or numpy.ndarray
            The position of the agent in the 2-dimensional space
        """
        if not isinstance(pos, (tuple, list, np.ndarray)):
            raise ValueError('pos must be a tuple, list, or nparray.')

        self.pos = pos
        self.sensory_inputs = self._sensory_inputs()
        return self.sensory_inputs