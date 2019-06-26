#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
from Layers import *
from Stimuli import *
from itertools import permutations
import numpy as np
import Checking

class Environment:
    def __init__(self, olf, gus, stims):
        """
        Parameters
        ----------
        olf: Layers.LiHopfield
            Innate connections among ORNs and primary olfactory interneurons
        gus: Layers.Single
            Innate connections among GRNs and primary gustatory interneurons
        stims: Stimuli.Stimuli
            The stimuli in the space
        """

        #================== Argument Check ============================
        Checking.arg_check(olf, 'olf', LiHopfield)
        Checking.arg_check(gus, 'gus', Single)
        #==============================================================

        # olf system
        self.__olf = olf
        # gus system
        self.__gus = gus
        # stimuli
        self.__stims = stims
        # the outer shape of the network
        self.__shape = np.array(stims.num_att())
        # the inner associative network
        self.__inner = BAM(self.__shape)
        # the kd tree for searching the stimuli
        self.__kd = LazyKDTree(stims)
        # the space limit (upper & right bounds)
        self.__lim = stims.get_max_pos()

        # the dimension of the array in each pixel
        self.__dim = self.__shape.sum()

        # build static stimulus environment
        self.__space = np.empty((self.__lim, self.__lim, self.__dim))
        # indices
        r = list(range(self.__lim))
        for i, j in permutations(r, 2):
            self.__space[i, j] = stims.odor_taste((i, j)) # assign to each pixel

        # ignore trivial values
        self.__space[np.isnan(self.__space)] = 0
        self.__space[self.__space < 1e-5] = 0

        # current position of the agent
        self.__pos = None
        # the nearby stimulus source
        self.__near = None
        # the distance to the nearby source
        self.__dist_to_near = 0

    def set_pos(self, pos, jump=False):
        """ Set the position of the agent, and feed the agent with the local
        sensory information

        Parameters
        ----------
        pos: array-like
            The targeted position
        jump: boolean
            If True, jump to another side of the space when pos is out of bounds
        """

        #================== Argument Check ============================
        if not jump:
            Checking.is_within(pos[0], (0, self.__lim))
            Checking.is_within(pos[1], (0, self.__lim))
        #==============================================================

        self.__pos = np.array(pos) % self.__lim
        self.__near, self.__dist_to_near = self.__kd.near(self.__pos)

    def get_pos(self):
        return self.__pos.copy()

    def sim(self, pos=(0, 0), epoch=10000, _eap=False):
        """ Run simulation

        Parameters
        ----------
        pos: array-like
            The targeted position
        epoch: int
            The maximum number of iterations
        _eap: boolean
            Enable printing details
        """

        self.set_pos(pos)

        for i in range(epoch + 1):
            if _eap: # enable printing
                print('Epoch {}:'.format(i))
                print('Currently at ', self.__pos)
                print('Approaching ', self.__near)

            self.__perceive(_eap)

            diff = self.__near - self.__pos

            if diff[0] == 0: # same x
                increment = (0, np.sign(diff[1]))
            elif diff[1] == 0: # same y
                increment = (np.sign(diff[0]), 0)
            else:
                slope = diff[1] / diff[0]
                sgn = np.sign(diff[0])
                increment = (sgn, sgn * slope)

            # while not reached
            while np.linalg.norm(self.__near - self.__pos) >= 1:
                self.__pos = self.__pos + increment
                if _eap:
                    print('Currently at ', self.__pos)
                self.__perceive(_eap)

            self.set_pos(self.__pos + 1, True) # leave the current point

    def __perceive(self, _eap):
        """ Feed the agent with both olf and gus stimuli. If gus stimulus is not
            all zeros, it will learn; otherwise, it will recall.
        """
        x, y = self.__get_stim()

        if _eap:
            print('Current environmental stimuli:\n\tOlf: {}\n\tGus: {}'\
                  .format(x, y))

        if (y == 0).all():
            pr = self.__inner.recall(self.__olf.feed(x))
            print('Predicting gus: {}'.format(pr))
            if True in np.isnan(pr):
                raise RuntimeError('NaN encountered during computation.')
        else:
            self.__inner.learn(self.__olf.feed(x), self.__gus.feed(y))

    def __get_stim(self):
        # round pos
        x, y = np.rint(self.__pos) % self.__lim
        x, y = int(x), int(y)

        return self.__space[x, y, :self.__shape[0]], \
                self.__space[x, y, self.__shape[0]:]