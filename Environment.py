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
        olf: Layers.Innate
            Innate connections among ORNs and primary olfactory interneurons
        gus: Layers.Innate
            Innate connections among GRNs and primary gustatory interneurons
        stims: Stimuli.Stimuli
            The stimuli in the space
        """

        #================== Argument Check ============================
        Checking.arg_check(olf, 'olf', Innate)
        Checking.arg_check(gus, 'gus', Innate)
        #==============================================================

        self.__olf = olf
        self.__gus = gus
        self.__stims = stims

        self.__kd = LazyKDTree(stims)

        self.__space_limit = stims.get_max_pos()
        self.__num_olf_att, self.__num_gus_att = stims.num_att()
        self.__dim = self.__num_olf_att + self.__num_gus_att

        # The attributes in each pixel is static, so initialize it
        self.__space = np.empty((self.__space_limit, self.__space_limit,
                                 self.__dim))

        rang = list(range(self.__space_limit))
        for i, j in permutations(rang, 2):
            self.__space[i, j] = stims.odor_taste((i, j))

        self.__pos = None
        self.__near = None
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
            Checking.is_within(pos[0], (0, self.__space_limit))
            Checking.is_within(pos[1], (0, self.__space_limit))
        #==============================================================

        self.__pos = np.array(pos) % self.__space_limit
        self.__near, self.__dist_to_near = self.__kd.near(self.__pos)

        x, y = int(self.__pos[0]), int(self.__pos[1])
        self.__olf.feed(self.__space[x, y, :self.__num_olf_att])
        self.__gus.feed(self.__space[x, y, self.__num_olf_att:])

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

        action = lambda a: round(a)

        for i in range(epoch + 1):
            if _eap:
                print('Epoch {}:'.format(i))
                print('Currently at ', self.__pos)
                print('Approaching ', self.__near)
            diff = self.__near - self.__pos
            if diff[0] == 0:
                increment = (0, np.sign(diff[1]))
            elif diff[1] == 0:
                increment = (np.sign(diff[0]), 0)
            else:
                slope = diff[1] / diff[0]
                sgn = np.sign(diff[0])
                increment = (sgn, sgn * slope)

            while np.linalg.norm(self.__near - self.__pos) >= 1:
                self.__pos = self.__pos + increment
                self.__feed()

            self.set_pos(self.__pos + 1, True) # leave the current point

    def __feed(self):
        x, y = np.rint(self.__pos) % self.__space_limit
        x, y = int(x), int(y)
        self.__olf.feed(self.__space[x, y, :self.__num_olf_att])
        self.__gus.feed(self.__space[x, y, self.__num_olf_att:])