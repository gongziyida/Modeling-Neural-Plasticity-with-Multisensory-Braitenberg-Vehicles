#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION
'''
import BraitenbergVehicles as bv
import Space as s
import Checking
import matplotlib.pyplot as plt
import networkx as nx

class Simulation:
    def __init__(self, BV, space):
        """
        Parameters
        ----------
        BV: BraitenbergVehicles.SingleSensor
            The agent to put in the space
        space: Space.Space
            The space where the agent explores
        """

        #================== Argument Check ============================
        Checking.arg_check(BV, 'BV', bv.SingleSensor)
        Checking.arg_check(space, 'space', s.Space)
        #==============================================================

        # the agent
        self._BV = BV
        # the space
        self._s = space
        # number of receptors
        self._num_orn, self._num_grn = self._s.get_num_receptors()

    def _train(self):
        att = self._s.get_stim_att()
        for stim in att:
            self._BV.feed(stim[:self._num_orn], stim[self._num_orn:], 'train')

    def _test(self, epoch):
        for i in range(epoch):
            pos = self._BV.get_pos()
            self._BV.feed(self._s.stim_at(pos), 'test')

    def _real(self, epoch, th):
        for i in range(epoch[0]):
            pos = self._BV.get_pos()
            target = self._s.near(pos)[0]
            self._BV.set_target(target)
            for j in range(epoch[1]):
                pos = self._BV.get_pos()
                olf, gus = self._s.stim_at(pos)
                self._BV.walk(gus)
                self._BV.feed(olf, gus, 'real')


    def sim(self, mode='real', epoch=None, th=0):
        if mode == 'train':
            self._train()
        elif mode == 'test':
            epoch = 1000 if epoch is None else epoch
            self._test(epoch)
        elif mode == 'real':
            epoch = [1000, 100] if epoch is None else epoch
            self._real(epoch, th)
        else:
            raise TypeError('The mode "'+ mode +'" is not understood.')
