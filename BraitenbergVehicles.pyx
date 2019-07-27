import numpy as np
cimport numpy as np
import Layers as l
import Movement as m
import networkx as nx
import matplotlib.pyplot as plt


cdef class SingleSensor:
    cdef object _olf, _gus, _asso, _motor, _pfunc

    def __init__(self, int orn_size, int grn_size, int gus_T, int lim, 
                 object pfuncs, int smell_period=50, int smell_tau=2.0, 
                 double olf_adapting_rate=0.0005,
                 double asso_adapting_rate=0.001, 
                 double asso_depression_rate=1e-10,
                 (int, int) init_pos=(0, 0)):

        self._olf = l.LiHopfield(orn_size, name='Olfaction',
                                 period=smell_period, tau=smell_tau,
                                 adapting_rate=olf_adapting_rate)

        self._gus = l.Single(grn_size, name='Gustation', act_func=lambda x: x)

        self._asso = l.BAM((orn_size, grn_size), name='Association',
                           adapting_rate=asso_adapting_rate,
                           depression_rate=asso_depression_rate)

        self._motor = m.RadMotor(grn_size, lim, pos=init_pos)

        self._pfunc = pfuncs


    def get_pos(self):
        return self._motor.get_pos()

    def set_target(self, (int, int) target):
        self._motor.heading(target)

    def is_at(self, (int, int) pos, double th=0.0):
        return self._motor.is_at(pos)

    cdef void _learn(self, np.ndarray[np.float_t] I1, 
                           np.ndarray[np.float_t] gus):
        cdef np.ndarray[np.float_t] I2
        I2 = self._gus.feed(gus)
        self._asso.learn(I1, I2)


    def np.ndarray[np.float_t] feed(self, np.ndarray[np.float_t] olf, 
                                         np.ndarray[np.float_t] gus, 
                                         str mode='real'):
        cdef np.ndarray[np.float_t] I1

        I1 = self._olf.feed(olf)

        if mode == 'real':
            if (gus == 0).all():
                return self._asso.recall(I1)
            else:
                self._learn(I1, gus)

        elif mode == 'train':
            self._learn(I1, gus)

        elif mode == 'test':
            return self._asso.recall(I1)

        else:
            raise TypeError('The mode "'+ mode +'" is not understood.')



    def double judge(self, np.ndarray[np.float_t] I):
        cdef double preference = 0.0
        cdef int i

        for i in range(I.shape[0]):
            preference += self._pfunc[i](I[i])

        return preference

    def void walk(self, np.ndarray[np.float_t] I):
        self._motor.set_preference(self.judge(I))
        self._motor.move()


    def save_network_img(self, fnames=None):
        if fnames is None:
            fnames = ('network_outer_struct.png',
                      'olf_unit.png', 'inner.png')

        # draw the outer structure
        self.save_network_outer_img(fnames[0])

        # draw olfactory bulb unit
        self._olf.save_img(fname=fnames[1])

        # draw the inner part
        self._asso.save_img(fname=fnames[2])

    def save_network_outer_img(self, fname='network_outer_struct.png'):
        g = nx.MultiDiGraph()

        # num of olf / gus attributes
        num_o, num_g = self._olf.shape()[0], self._gus.shape()[0]

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