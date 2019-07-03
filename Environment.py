#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION
Need supports: Networkx, Numpy, Scipy, Matplotlib, pygraphviz, graphviz
environment: sudo apt install graphviz libgraphviz-dev pkg-config
'''
from Layers import *
from Stimuli import *
from itertools import product
import numpy as np
import Checking
import matplotlib.pyplot as plt
import networkx as nx

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
        # stim pos for printing
        self.__stims_pos = stims.get_pos().T
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
        for i, j in product(r, r):
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

    def save_space_img(self, name_prefices=('odor_space', 'taste_space')):
        for i in (0, 1):
            # the largest pixel value
            if i == 0:
                vmax = self.__space[:, :, :self.__shape[0]].max()
                init, end = 0, self.__shape[0]
            else:
                vmax = self.__space[:, :, self.__shape[0]:].max()
                init, end = self.__shape

            for k in range(end):
                plt.clf()
                plt.rcParams["figure.figsize"] = [8, 8]

                fname = name_prefices[i] + '_{}.png'.format(k) # file name

                # heatmap
                im = plt.imshow(self.__space[:, :, init + k].T,
                                vmin=0, vmax=vmax, origin='lower')
                plt.colorbar(im, fraction=0.02, pad=0.01)

                if init == 0:
                    # stimulus source locations
                    plt.scatter(*self.__stims_pos, s=2, c='r')

                plt.axis('off') # no need for axes
                plt.savefig(fname, dpi=100, bbox_inches='tight',
                            transparent=True)

    def save_network_img(self, fname='network_outer_struct.png'):
        g = nx.MultiDiGraph()

        # num of olf / gus attributes
        num_o, num_g = self.__shape

        # node -1: olfactory bulb unit
        # node -2: preference unit
        # node 0 ~ 9: olfactory inputs
        # node 10 ~ 19: olfactory interneurons
        # node 20 ~ 24: gustatory interneurons
        # node 25 ~ 29: gustatory inputs

        # add edges
        for i in range(num_o):
            g.add_edge(i, -1, color='black')
            g.add_edge(-1, i + num_o, color='black')


        # define fixed positions of olfaction-related nodes
        fixed_pos = {}
        for i in range(num_o):
            p = i - num_o / 2 # position of node
            fixed_pos.update({i: (-1, p), i + num_o: (1, p)})

        # define fixed positions of gustation-related nodes
        for i in range(num_o * 2, num_o * 2 + num_g):
            for j in range(num_o, num_o * 2):
                g.add_edge(j, i, color='gray') # hebbian synapses
            g.add_edge(i + num_g, i, color='black') # inputs
            g.add_edge(i, -2, color='black') # to preference unit
            p = i - (num_o * 2 + num_g / 2) # position of node
            fixed_pos.update({i: (2, p), i + num_g: (3, p)})

        fixed_pos.update({-1: (0, 0), -2: (3, -4)})

        # Extract a list of edge colors
        edges = g.edges()
        edge_colors = [g[u][v][0]['color'] for u, v in edges]

        # Extract a list of node colors
        node_colors = ['gray' if node < 0 else 'black' for node in g]

        # Extract a list of node size
        node_sizes = [2000 if node < 0 else 300 for node in g]

        # get the nx positions
        pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

        # draw outer structure
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 8)
        nx.draw(g, pos=pos, node_color=node_colors, edge_color=edge_colors,
                node_size=node_sizes, ax=ax)
        fig.savefig(fname, dpi=100, bbox_inches='tight', transparent=True)

        # draw olfactory bulb unit
        self.__olf.save_img(fname='olf_unit.png')


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
