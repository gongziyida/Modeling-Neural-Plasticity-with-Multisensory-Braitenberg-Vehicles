#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION: 0.1
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np
import matplotlib.pyplot as plt
import Checking
from itertools import product

class Space:
    def __init__(self, num_stim, num_orn, num_grn, mapping, method='random',
                 max_pos=500, gus_T=2):
        """
        Parameters
        ----------
        num_stim: int
            The number of stimuli to be initialized
        num_orn: int
            The number of olfactory attributes in a stimulus
        num_grn: int
            The number of gustatory attributes in a stimulus
        mapping: collection of callables
            The collection of mapping types, from olfactory attributes to
            gustatory attributes
        method: str
            The method to choose the locations of stimulus sources. Default is
            'random'. Another option can be 'matrix'.
        max_pos: int
            The maximum value in each axis
        gus_T: int
            The threshold within which the gustatory information is detectable
        """

        #================== Argument Check ============================
        Checking.arg_check(mapping, 'mapping', 'callable')
        #==============================================================

        self._max_pos = max_pos
        self._gus_T = gus_T
        self._num_stim = num_stim
        self._num_orn = num_orn
        self._num_grn = num_grn
        self._pixel_dim = num_orn + num_grn

        self._pos = self._set_stim_pos(method)

        self._att = np.empty((num_stim, self._pixel_dim))
        self._att[:, :num_orn] = \
            np.abs(np.random.normal(size=(num_stim, num_orn)))

        for i in range(num_stim):
            self._att[i, num_orn:] = mapping(self._att[i, :num_orn])

        # the kd tree for searching the stimuli
        self._kd = LazyKDTree(self._pos, max_pos)

        # build static stimulus environment
        self._space = self._build_space()


    def _set_stim_pos(self, method):
        if method == 'random':
            pos = list(range(self._max_pos))
            pos = np.array(list(product(pos, pos)))

            max_index = self._max_pos**2

            indices = np.random.choice(range(max_index),
                                       self._num_stim, replace=False)
            return pos[indices]

        elif method == 'matrix':
            nrows = int(np.sqrt(self._num_stim))
            ncols = self._num_stim // nrows
            rmd = self._num_stim - nrows * ncols

            if rmd != 0:
                i = rmd // ncols
                nrows += i + 1
                rmd -= i * ncols

            xitvl = int(np.ceil(self._max_pos / ncols))
            yitvl = int(np.ceil(self._max_pos / nrows))

            xs = list(range(0, self._max_pos, xitvl))
            ys = list(range(0, self._max_pos, yitvl))
            pos = np.array(list(product(xs, ys)))
            if rmd == 0:
                return pos
            else:
                return pos[:rmd - ncols]


    def _build_space(self):
        s = np.empty((self._max_pos, self._max_pos, self._pixel_dim))
        # indices
        r = list(range(self._max_pos))
        for i, j in product(r, r):
            s[i, j] = self._attr_at((i, j)) # assign to each pixel

        # ignore trivial values
        s[np.isnan(s)] = 0
        s[s < 1e-5] = 0

        return s

    def _dist_to(self, pos):
        diff = np.abs(pos - self._pos)
        # when the agent hits a boundary, it continues on the other side.
        # Therefore, the differences in coordinates from both directions are
        # considered.
        to_switch = diff > self._max_pos / 2
        diff[to_switch] = self._max_pos - diff[to_switch]

        return np.linalg.norm(diff, axis=1)

    def _attr_at(self, pos):
        dist = self._dist_to(pos)
        factor = np.exp(- dist / self._max_pos * 10)
        odor = factor @ self._att[:, :self._num_orn]

        indices = np.argwhere(dist < self._gus_T)[:, 0]
        taste = self._att[indices, self._num_orn:]
        taste = np.sum(taste, axis=0)

        ot = np.append(odor, taste)
        return ot

    def size(self):
        return self._num_stim

    def get_num_receptors(self):
        return self._num_orn, self._num_grn

    def get_stim_pos(self):
        return self._pos.copy()

    def get_stim_att(self):
        return self._att.copy()

    def get_max_pos(self):
        return self._max_pos

    def stim_at(self, pos):
        # round pos
        x, y = np.rint(pos) % self._max_pos
        x, y = int(x), int(y)

        return self._space[x, y, :self._num_orn], \
                self._space[x, y, self._num_orn:]

    def near(self, pos):
        return self._kd.near(pos)

    def save_img(self, name_prefices=('odor_space', 'taste_space')):
        for i in (0, 1):
            # the largest pixel value
            if i == 0:
                vmax = self._space[:, :, :self._num_orn].max()
                init, end = 0, self._num_orn
            else:
                vmax = self._space[:, :, self._num_orn:].max()
                init, end = self._num_orn, self._num_grn

            for k in range(end):
                plt.clf()
                plt.rcParams["figure.figsize"] = [8, 8]

                fname = name_prefices[i] + '_{}.png'.format(k) # file name

                # heatmap
                im = plt.imshow(self._space[:, :, init + k].T,
                                vmin=0, vmax=vmax, origin='lower')
                plt.colorbar(im, fraction=0.02, pad=0.01)

                if init == 0:
                    # stimulus source locations
                    plt.scatter(*self._pos.T, s=5, c='r')

                plt.axis('off') # no need for axes
                plt.savefig(fname, dpi=100, bbox_inches='tight',
                            transparent=True)


class LazyKDTree:
    class Node:
        def __init__(self, pos, lc, rc):
            self.pos = pos
            self.lc = lc
            self.rc = rc
            self.flag = False


        def is_leaf(self):
            return (self.lc is None) and (self.rc is None)

    def __init__(self, pos, max_pos):
        """
        Parameters
        ----------
        pos: numpy.ndarray
            The positions of the stimuli to feed in this KD Tree
        max_pos: int
            The maximum value in each axis
        """

        #================== Argument Check ============================
        Checking.arg_check(pos, 'pos', np.ndarray)
        #==============================================================
        self._num_stim = pos.shape[0]
        self._max_pos = max_pos

        sort = sorted(pos, key=lambda s: s[0])

        self._tree = self._build(sort, 0)

        self._num_visited = 0
        self._flag = True


    def _build(self, sort, dim):
        rang = len(sort)
        if rang == 1:
            return LazyKDTree.Node(sort[0], None, None)
        elif rang == 0:
            return None
        center = rang // 2
        val = sort[center]

        dim = 1 - dim # next dimension
        key = lambda s: s[dim]
        lc = self._build(sorted(sort[: center], key=key), dim)
        rc = self._build(sorted(sort[center + 1:], key=key), dim)

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
        cur = self._tree

        local_min = [cur, self._dist_to(cur, pos)] # note down the local minimum

        self._near(pos, cur, local_min, 0, _eap)

        self._num_visited += 1
        if self._num_visited == self._num_stim:
            self._num_visited = 0
            self._flag = not self._flag

        local_min[0].flag = self._flag

        return local_min[0].pos, local_min[1]

    def _near(self, pos, cur, local_min, dim, _eap):
        to_cur = self._dist_to(cur, pos)

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
            self._near(pos, first, local_min, 1 - dim, _eap)
        # reaching the end; about to backtrack
        elif cur.flag != self._flag:
            local_min[0], local_min[1] = cur, to_cur

        # backtracking
        if to_cur < local_min[1] and  cur.flag != self._flag:
            local_min[0], local_min[1] = cur, to_cur

        if not (second is None):
            to_line = abs(pos[dim] - second.pos[dim])
            if to_line < local_min[1]:
                self._near(pos, second, local_min, 1 - dim, _eap)

    def _dist_to(self, n, pos):
        diff = np.abs(pos - n.pos)
        # when the agent hits a boundary, it continues on the other side.
        # Therefore, the differences in coordinates from both directions are
        # considered.
        to_switch = diff > self._max_pos / 2
        diff[to_switch] = self._max_pos - diff[to_switch]

        return np.linalg.norm(diff)