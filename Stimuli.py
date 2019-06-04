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

class Stimuli:
    def __init__(self, num_stim, num_olf_att, num_gus_att, att_mappings,
                 max_pos=1000, gus_threshold=5):
       self.__max_pos = max_pos
       self.__gus_threshold = gus_threshold
       self.__num_stim = num_stim
       self.__num_olf_att = num_olf_att
       self.__num_gus_att = num_gus_att

       self.__pos = np.random.randint(max_pos, size=(num_stim, 2))

       self.__num_mappings = len(att_mappings)

       self.__att = np.empty((num_stim, num_olf_att + num_gus_att))
       self.__att[:, :num_olf_att] = \
           np.random.normal(size=(num_stim, num_olf_att))

       for i in range(num_stim):
           mapping = np.random.choice(att_mappings)
           self.__att[i, num_olf_att:] = mapping(self.__att[i, :num_olf_att])


    def dist_to(self, pos):
        return np.linalg.norm(pos - self.__pos, axis=1)

    def odor_taste(self, dist):
        factor = np.exp(- dist / self.__max_pos * 10)
        odor = factor @ self.__att[:, :self.__num_olf_att]

        indices = np.argwhere(dist < self.__gus_threshold)[:, 0]
        taste = np.sum(self.__att[indices, self.__num_olf_att], axis=0)

        return np.append(odor, taste, axis=1)

    def size(self):
        return self.__num_stim

    def get_pos(self):
        return self.__pos.copy()



class LazyKDTree:
    class Node:
        def __init__(self, pos, lc, rc):
            self.pos = pos
            self.lc = lc
            self.rc = rc

        def dist_to(self, pos):
            return np.linalg.norm(pos - self.pos)

        def is_leaf(self):
            return (self.lc is None) and (self.rc is None)

    def __init__(self, stim):
        #================== Argument Check ============================
        Checking.arg_check(stim, 'stim', Stimuli)
        #==============================================================
        self.__num_stim = stim.size()

        pos = stim.get_pos()
        sort = sorted(pos, key=lambda s: s[0])

        self.__tree = self.__build(sort, 0)

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
        cur = self.__tree

        local_min = [None, np.inf] # note down the local minimum

        self.__near(pos, cur, local_min, 0, _eap)

        return local_min

    def __near(self, pos, cur, local_min, dim, _eap):
        to_cur = cur.dist_to(pos)

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
        else: # reaching the end; about to backtrack
            local_min[0], local_min[1] = cur.pos, to_cur

        if to_cur < local_min[1]: # backtracking
            local_min[0], local_min[1] = cur.pos, to_cur

        if not (second is None):
            to_line = abs(pos[dim] - second.pos[dim])
            if to_line < local_min[1]:
                self.__near(pos, second, local_min, 1 - dim, _eap)
