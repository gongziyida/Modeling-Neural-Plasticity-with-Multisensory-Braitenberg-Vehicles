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
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

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
        self._fig, self._ax = plt.subplots(2)
        self._fig.set_size_inches(10, 20)

        max_pos = self._s.get_max_pos()
        self._ax[0].set_xlim(0, max_pos)
        self._ax[0].set_ylim(0, max_pos)
        self._agent = self._ax[0].scatter(*self._BV.get_pos())
        self._agent.set_offset_position('data')

        xlabels = ['Odor {}'.format(i) for i in range(1, 11)]
        xlabels += ['Taste {}'.format(i) for i in range(1, 6)]
        xlabels += ['Preference']
        self._sensors = self._ax[1].bar(xlabels, [0] * len(xlabels))
        self._ax[1].set_ylim([-5, 5])

        self._ani = FuncAnimation(self._fig, self._update, interval=10,
                                  frames=self._frame(epoch), save_count=epoch)
        self._ani.save('animation.gif', writer='pillow')

    def _frame(self, epoch):
        for i in range(epoch):
            if i % 10 == 0:
                pos = self._BV.get_pos()
                target = self._s.near(pos)[0]
                self._BV.set_target(target)
            pos = self._BV.get_pos()
            olf, gus = self._s.stim_at(pos)
            yield pos, np.append(np.append(olf, gus), self._BV.judge(gus))
            self._BV.walk(gus)
            self._BV.feed(olf, gus, 'real')

    def _update(self, arg):
        self._agent.set_offsets(arg[0])
        for bar, h in zip(self._sensors, arg[1]):
            bar.set_height(h)


    def sim(self, mode='real', epoch=1000, th=0):
        if mode == 'train':
            self._train()
        elif mode == 'test':
            self._test(epoch)
        elif mode == 'real':
            self._real(epoch, th)
        else:
            raise TypeError('The mode "'+ mode +'" is not understood.')
