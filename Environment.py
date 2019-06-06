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

    def set_pos(self, pos):
        x, y = pos
        x, y = int(x), int(y)
        self.__pos = (x, y)
        self.__near, self.__dist_to_near = self.__kd.near(self.__pos)
        self.__olf.feed(self.__space[x, y, :self.__num_olf_att])
        self.__gus.feed(self.__space[x, y, self.__num_olf_att:])

    def sim(self, pos=(0, 0), epoch=10000):
        self.set_pos(pos)
        diff = self.__near - self.__pos
        slope = diff[1] / diff[0]
        for i in range(epoch):
            while not (self.__pos == self.__near).all():
                self.set_pos(self.__pos + (1, slope))
            self.__set_pos(self.__pos + 1)