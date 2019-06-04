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

class Stimulus:
    def __init__(self, olf_att, gus_att, pos):
        #================== Argument Check ============================
        Checking.arg_check(olf_att, 'olf_att', np.ndarray)
        Checking.arg_check(gus_att, 'gus_att', np.ndarray)
        Checking.arg_check(pos, 'pos', np.ndarray)
        #==============================================================

        self.olf_att = olf_att
        self.gus_att = gus_att
        self.__pos = pos
        self.__pos.setflags(write=False)

    def pos(self):
        return self.__pos

    def dist_to(self, pos):
        return np.linalg.norm(pos - self.__pos)

class KDTree:
    class Node:
        def __init__(self, stim, lc, rc):
            self.stim = stim
            self.lc = lc
            self.rc = rc

        def dist_to(self, pos):
            return self.stim.dist_to(pos)

        def is_leaf(self):
            return (self.lc is None) and (self.rc is None)

    def __init__(self, stims):
        #================== Argument Check ============================
        Checking.arg_check(stims, 'stims', list)
        #==============================================================

        self.__num_stim = len(stims)

        sort = sorted(stims, key=lambda s: s.pos()[0])

        self.__tree = self.__build(sort, 0)

    def __build(self, sort, dim):
        rang = len(sort)
        if rang == 1:
            return KDTree.Node(sort[0], None, None)
        elif rang == 0:
            return None
        center = rang // 2
        val = sort[center]

        dim = 1 - dim # next dimension
        key = lambda s: s.pos()[dim]
        lc = self.__build(sorted(sort[: center], key=key), dim)
        rc = self.__build(sorted(sort[center + 1 :], key=key), dim)

        return KDTree.Node(val, lc, rc)

    def near(self, pos, _eap=False):
        cur = self.__tree

        local_min = [None, np.inf] # note down the local minimum

        self.__near(pos, cur, local_min, 0, _eap)

        return local_min

    def __near(self, pos, cur, local_min, dim, _eap):
        to_cur = cur.dist_to(pos)

        if _eap: # enable printing traversal processes
            print('current: ', cur.stim.pos())
            print('lc: ', 'X' if cur.lc is None else cur.lc.stim.pos())
            print('rc: ', 'X' if cur.rc is None else cur.rc.stim.pos())
            print('current local min: ', local_min[1])
            print('---')

        if pos[dim] < cur.stim.pos()[dim]:
            first, second = cur.lc, cur.rc
        else:
            second, first = cur.lc, cur.rc

        if not (first is None):
            self.__near(pos, first, local_min, 1 - dim, _eap)
        else: # reaching the end; about to backtrack
            local_min[0], local_min[1] = cur.stim, to_cur

        if to_cur < local_min[1]: # backtracking
            local_min[0], local_min[1] = cur.stim, to_cur

        if not (second is None):
            to_line = abs(pos[dim] - second.stim.pos()[dim])
            if to_line < local_min[1]:
                self.__near(pos, second, local_min, 1 - dim, _eap)
