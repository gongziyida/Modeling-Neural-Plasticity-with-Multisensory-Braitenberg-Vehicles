#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np

class Motor:
    def __init__(self, size, lim, minStep=1, pos=(0, 0)):
        self._size = size
        self._lim = lim
        self._minStep = minStep
        self._pos = np.array(pos, dtype='float64')
        self._increments = np.zeros(2, dtype='float64')
        self._preference = 0

    def get_pos(self):
        return self._pos.copy()

    def is_at(self, target, th=0):
        dist = np.linalg.norm(self._pos - target)
        return dist <= th


class RadMotor(Motor):
    def __init__(self, size, lim, headingRad=0, minStep=1, pos=(0, 0)):
        super().__init__(size, lim, minStep, pos)
        # the heading angle in radian
        self._headingRad = headingRad
        # the previous preference
        self._prev_preference = 0
        # the direction of the target, in radian
        self._target_dir = None


    def get_headingRad(self):
        return self._headingRad

    def get_increments(self, inplace=True):
        x = self._minStep * np.cos(self._headingRad)
        y = self._minStep * np.sin(self._headingRad)

        if inplace:
            self._increments = np.array((x, y))
        else:
            return self._increments.copy()

    def set_preference(self, p):
        self._prev_preference = self._preference
        self._preference = p

    def _round_rad(self):
        self._headingRad %= np.pi * 2

    def _decide(self):
        p = self._preference - self._prev_preference

        if p > 0:
            return np.random.normal(self._headingRad, p / np.pi)
        elif p == 0:
            instinct = np.random.randint(-100, 100) / 100 * np.pi
            return np.random.choice((instinct, self._target_dir))
        else:
            return np.random.normal(np.pi + self._headingRad, np.abs(p) / np.pi)


    def move(self):
        # make decision on where to go next
        self._headingRad = self._decide()

        self.get_increments()

        # actually move
        self._pos += self._increments
        self._pos %= self._lim # do not exceed the boundary


    def rotate(self, rad):
        self._headingRad += rad
        self._round_rad()


    def heading(self, target):
        xdiff, ydiff = self._pos - target
        self._headingRad = np.arctan(ydiff / xdiff)
        self._round_rad()
        self._target_dir = self._headingRad
