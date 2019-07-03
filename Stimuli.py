#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION: 0.1
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np
import Checking
from itertools import product

class Stimuli:
    def __init__(self, num_stim, num_olf_att, num_gus_att, mapping,
                 max_pos=1000, gus_threshold=3):
        """
        Parameters
        ----------
        num_stim: int
            The number of stimuli to be initialized
        num_olf_att: int
            The number of olfactory attributes in a stimulus
        num_gus_att: int
            The number of gustatory attributes in a stimulus
        mapping: collection of callables
            The collection of mapping types, from olfactory attributes to
            gustatory attributes
        max_pos: int
            The maximum position (x or y) value
        gus_threshold: float
            The threshold within which the gustatory information is detectable
        """

        #================== Argument Check ============================
        Checking.arg_check(mapping, 'mapping', 'callable')
        #==============================================================

        self.__max_pos = int(max_pos)
        self.__gus_threshold = float(gus_threshold)
        self.__num_stim = int(num_stim)
        self.__num_olf_att = int(num_olf_att)
        self.__num_gus_att = int(num_gus_att)

        pos = list(range(max_pos))
        pos = np.array(list(product(pos, pos)))
        indices = np.random.choice(range(max_pos**2), num_stim, replace=False)
        self.__pos = pos[indices]

        self.__att = np.empty((num_stim, num_olf_att + num_gus_att))
        self.__att[:, :num_olf_att] = \
            np.abs(np.random.normal(size=(num_stim, num_olf_att)))

        for i in range(num_stim):
            self.__att[i, num_olf_att:] = mapping(self.__att[i, :num_olf_att])


    def dist_to(self, pos):
        diff = np.abs(pos - self.__pos)
        # when the agent hits a boundary, it continues on the other side.
        # Therefore, the differences in coordinates from both directions are
        # considered.
        to_switch = diff > self.__max_pos / 2
        diff[to_switch] = self.__max_pos - diff[to_switch]

        return np.linalg.norm(diff, axis=1)

    def odor_taste(self, pos):
        dist = self.dist_to(pos)
        factor = np.exp(- dist / self.__max_pos * 10)
        odor = factor @ self.__att[:, :self.__num_olf_att]

        indices = np.argwhere(dist < self.__gus_threshold)[:, 0]
        taste = self.__att[indices, self.__num_olf_att:]
        taste = np.sum(taste, axis=0)

        ot = np.append(odor, taste)
        return ot

    def size(self):
        return self.__num_stim

    def num_att(self):
        return (self.__num_olf_att, self.__num_gus_att)

    def get_pos(self):
        return self.__pos.copy()

    def get_max_pos(self):
        return self.__max_pos


class LazyKDTree:
    class Node:
        def __init__(self, pos, lc, rc):
            self.pos = pos
            self.lc = lc
            self.rc = rc
            self.flag = False


        def is_leaf(self):
            return (self.lc is None) and (self.rc is None)

    def __init__(self, stim):
        """
        Parameters
        ----------
        stim: Stimuli.Stimuli
            The stimuli to feed in this KD Tree
        """

        #================== Argument Check ============================
        Checking.arg_check(stim, 'stim', Stimuli)
        #==============================================================
        self.__num_stim = stim.size()

        pos = stim.get_pos()
        sort = sorted(pos, key=lambda s: s[0])

        self.__tree = self.__build(sort, 0)

        self.__num_visited = 0
        self.__flag = True

        self.__max_pos = stim.get_max_pos()

    def __build(self, sort, dim):
        rang = len(sort)
        if rang == 1:
            return LazyKDTree.Node(sort[0], None, None)
        elif rang == 0:
            return None
        center = rang // 2
        val = sort[center]

        dim = 1 - dim # next dimension
        key = lambda s: s[dim]
        lc = self.__build(sorted(sort[: center], key=key), dim)
        rc = self.__build(sorted(sort[center + 1:], key=key), dim)

        return LazyKDTree.Node(val, lc, rc)

    def near(self, pos, _eap=False):
        """ Find a nearby point

        Parameters
        ----------
        pos: array-like
            The position of the query point
        _eap: boolean
            Enable printing details

        Returns
        ----------
        pos_nearby: array-like
            The position of a nearby point
        dist: float
            The distance to this point
        """
        cur = self.__tree

        local_min = [cur, self.__dist_to(cur, pos)] # note down the local minimum

        self.__near(pos, cur, local_min, 0, _eap)

        self.__num_visited += 1
        if self.__num_visited == self.__num_stim:
            self.__num_visited = 0
            self.__flag = not self.__flag

        local_min[0].flag = self.__flag

        return local_min[0].pos, local_min[1]

    def __near(self, pos, cur, local_min, dim, _eap):
        to_cur = self.__dist_to(cur, pos)

        if _eap: # enable printing traversal processes
            print('current: ', cur.pos)
            print('lc: ', 'X' if cur.lc is None else cur.lc.pos)
            print('rc: ', 'X' if cur.rc is None else cur.rc.pos)
            print('current local min: ', local_min[1])
            print('---')

        if pos[dim] < cur.pos[dim]:
            first, second = cur.lc, cur.rc
        else:
            second, first = cur.lc, cur.rc

        if not (first is None):
            self.__near(pos, first, local_min, 1 - dim, _eap)
        # reaching the end; about to backtrack
        elif cur.flag != self.__flag:
            local_min[0], local_min[1] = cur, to_cur

        # backtracking
        if to_cur < local_min[1] and  cur.flag != self.__flag:
            local_min[0], local_min[1] = cur, to_cur

        if not (second is None):
            to_line = abs(pos[dim] - second.pos[dim])
            if to_line < local_min[1]:
                self.__near(pos, second, local_min, 1 - dim, _eap)

    def __dist_to(self, n, pos):
        diff = np.abs(pos - n.pos)
        # when the agent hits a boundary, it continues on the other side.
        # Therefore, the differences in coordinates from both directions are
        # considered.
        to_switch = diff > self.__max_pos / 2
        diff[to_switch] = self.__max_pos - diff[to_switch]

        return np.linalg.norm(diff)