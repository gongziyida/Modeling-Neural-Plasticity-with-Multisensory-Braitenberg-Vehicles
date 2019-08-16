#cython: language_level=3
cimport Space
import Space
cimport Layers
import Layers
cimport Movement
import Movement
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport abs as cabs
import networkx as nx
import matplotlib.pyplot as plt

cdef extern from "core.h":
    void judge(double *preference, double *to_judge, int *pfunc, int num_grn)

cdef class Simulation:

    def __init__(self, Layers.LiHopfield olf, Layers.BAM asso, Layers.Single gus, 
                    Movement.RadMotor m, Space.Space space, 
                    np.ndarray[np.int32_t] pfunc, object mapping,
                    Py_ssize_t max_num_pref_memorized=50):
        """
        Parameters
        ----------
        olf: Layers.LiHopfield

        asso: Layers.BAM
        
        gus: Layers.Single
        
        m: Movement.RadMotor
        
        space: Space.Space
        
        pfunc: np.ndarray[np.int32_t]
            an integer array consisting of integer IDs of preference functions
            0: positive linear correlation
            1: negative linear correlation
            2: gaussian with mu = 0.5, sigma = 0.05
        
        max_num_pref_memorized: Py_ssize_t
            the maximum number of preference values memorized
        """

        self._olf = olf
        self._asso = asso
        self._gus = gus
        self._m = m
        self._space = space

        o, g = self._space.get_num_receptors()
        self._num_orn, self._num_grn = <Py_ssize_t> o, <Py_ssize_t> g
        
        self._pos = np.zeros((1, 2), dtype=np.float64)

        self._stim = np.zeros(o+g, dtype=np.float64)

        self._I1 = np.zeros(o, dtype=np.float64)

        self._ideal_I2 = np.zeros(g, dtype=np.float64)

        self._gus_zeros = np.zeros(g, dtype=np.float64)
        self._recalled = self._gus_zeros

        self._pfunc = pfunc

        self._mapping = mapping

        self._preference = np.zeros(max_num_pref_memorized, dtype=np.float64)
        self._ideal_preference = self._preference.copy()

        self._mnpm = max_num_pref_memorized

        self._step_counter = 0


    cpdef void set_target(self):
        # get the current position
        pos = self._m.get_pos() 

        # find one stimulus source nearby
        target = self._space.near(pos)

        # change heading direction
        self._m.heading(target)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef void step(self):
        cdef Py_ssize_t i

        # get the current position
        self._pos[0] = self._m.get_pos() 

        # get the stimulus at the current position
        self._stim = self._space.stim_at(<Py_ssize_t> self._pos[0, 0], 
                                         <Py_ssize_t> self._pos[0, 1])


        # get the output values of the olfactory bulb
        self._I1 = self._olf.feed(self._stim[:self._num_orn])

        cdef double[::1] I2 # auxiliary variable for gus values

        # check if there is any gus value
        cdef bint allzero = True
        for i in range(self._num_grn):
            if self._stim[i+self._num_orn] != 0:
                allzero = False
                break

        # auxiliary variables
        cdef Py_ssize_t index = self._step_counter
        cdef double *to_judge

        # for avoiding obsession
        cdef int occurence = 0
        cdef int sig_ign = 0
        cdef double d1, d2, d3

        if self._step_counter >= self._mnpm: # need to shift
            index = self._mnpm - 1 # last index
            self._preference[:index] = self._preference[1:]
            self._preference[index] = 0 # clear

            self._ideal_preference[:index] = self._ideal_preference[1:]
            self._ideal_preference[index] = 0 # clear

            # help check if obsession occurs
            for i in range(0, self._mnpm - self._mnpm % 3, 3):
                d1 = cabs(self._preference[i] - self._preference[i + 1])
                d2 = cabs(self._preference[i] - self._preference[i + 2])
                d3 = cabs(self._preference[i + 1] - self._preference[i + 2])

                # if the differences between i & i+1, i & i+2, i+1 & i+2 are all small,
                # there should be a "plateau"
                # or if the difference between i & i+2 is small while the other two are not,
                # there should be a "saw"
                if (d1 < 1 and d2 < 1 and d3 < 1) or \
                    (d3 < 1 and d2 > 1 and d1 > 1):
                    occurence += 1


        if allzero: # if all are zero, recall from the memory
            self._recalled = self._asso.recall(self._I1)
            to_judge = &self._recalled[0]
            
        else: # learn
            I2 = self._gus.feed(self._stim[self._num_orn:])
            self._asso.learn(self._I1, I2)

            # nothing recalled, so set to default (zeros)
            self._recalled = self._gus_zeros

            to_judge = &self._stim[self._num_orn]
        
        # get the ideal gus
        self._mapping(self._stim, self._ideal_I2)

        # get preference based on gus values
        judge(&self._preference[index], to_judge, &self._pfunc[0], self._num_grn)
        judge(&self._ideal_preference[index], &self._ideal_I2[0], &self._pfunc[0], self._num_grn)
        
        # if the occurence (in which three preference values are compared) is about half 
        # the maximum number of perference memorized, ignorance signal will be set to 1
        if occurence > self._mnpm // (2 * 3):
            sig_ign = 1
        else:
            # preference goes to the motor system
            self._m.set_preference(self._preference[index])

        # update BV position
        self._pos[0] = self._m.move(sig_ign)


        self._step_counter += 1


    def get_BV_pos(self):
        return self._pos

    def get_pref(self):
        return self._preference

    def get_cur_stim(self):
        return self._stim

    def get_stim(self):
        return self._space.get_stim_pos()

    def get_max_pos(self):
        return self._space.get_max_pos()

    def get_num_receptors(self):
        return self._num_orn, self._num_grn

    def get_olf_bulb_output(self):
        return self._I1

    def get_asso_weight(self):
        return self._asso.get_weight()

    def get_recalled(self):
        return self._recalled

    def get_Gx_record(self):
        return self._olf.get_Gx_record()

    def get_max_num_pref_memorized(self):
        return self._mnpm

    def get_ideal_gus(self):
        return self._ideal_I2

    def get_ideal_pref(self):
        return self._ideal_preference


    def set_olf_param(self, **kwarg):
        if 'tau' in kwarg:
            self._olf.set_tau(kwarg['tau'])

        if 'adapting_rate' in kwarg:
            self._olf.set_adapting_rate(kwarg['adapting_rate'])


    def set_asso_param(self, **kwarg):
        if 'adapting_rate' in kwarg:
            self._asso.set_adapting_rate(kwarg['adapting_rate'])

        if 'depression_rate' in kwarg:
            self._asso.set_depression_rate(kwarg['depression_rate'])


    def save_network_outer_img(self, name='network_outer_struct.png'):
        g = nx.MultiDiGraph()

        # node -1: olfactory bulb unit
        # node -2: preference unit
        # node 0 ~ 9: olfactory inputs
        # node 10 ~ 19: olfactory interneurons
        # node 20 ~ 24: gustatory interneurons
        # node 25 ~ 29: gustatory inputs

        # add edges
        for i in range(self._num_orn):
            g.add_edge(i, -1, color='black')
            g.add_edge(-1, i + self._num_orn, color='black')


        # define fixed positions of olfaction-related nodes
        fixed_pos = {}
        for i in range(self._num_orn):
            p = i - self._num_orn / 2 # position of node
            fixed_pos.update({i: (-1, p), i + self._num_orn: (1, p)})

        # define fixed positions of gustation-related nodes
        for i in range(self._num_orn * 2, self._num_orn * 2 + self._num_grn):
            for j in range(self._num_orn, self._num_orn * 2):
                g.add_edge(j, i, color='gray') # hebbian synapses
            g.add_edge(i + self._num_grn, i, color='black') # inputs
            g.add_edge(i, -2, color='black') # to preference unit
            p = i - (self._num_orn * 2 + self._num_grn / 2) # position of node
            fixed_pos.update({i: (2, p), i + self._num_grn: (3, p)})

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
        fig.savefig(name, dpi=100, bbox_inches='tight', transparent=True)


class Example(Simulation):
    def __init__(self):
        gus = Layers.Single(5, lambda x: x)                                     
        
        olf = Layers.LiHopfield(10)                                             
        
        m = Movement.RadMotor(200)                                              
        
        asso = Layers.BAM(10, 5)                                                
        
        def mapping(x, y): 
            for i in range(x.shape[0]): 
                for j in range(x.shape[1]): 
                    y[i, j // 2] += x[i, j]
                y[i] /= np.linalg.norm(y[i])

        space = Space.Space(20, 10, 5, mapping)                                 

        pfunc = np.array([0, 0, 1, 1, 2], dtype=np.int32)                       

        super().__init__(olf, asso, gus, m, space, pfunc)