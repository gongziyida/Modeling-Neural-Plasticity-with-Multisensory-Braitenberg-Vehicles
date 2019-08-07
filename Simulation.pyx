#cython: language_level=3
cimport Space
import Space
cimport Layers
import Layers
cimport Movement
import Movement
cimport numpy as np
import numpy as np

cdef extern from "core.h":
    void judge(double *preference, double *to_judge, int *pfunc, int num_grn)

cdef class Simulation:
    cdef Layers.LiHopfield _olf
    cdef Layers.BAM _asso
    cdef Layers.Single _gus 
    cdef Movement.RadMotor _m
    cdef Space.Space _space
    cdef double _preference
    cdef int[::1] _pfunc
    cdef double[::1] _stim, _I1, _recalled, _recalled_default
    cdef double[:,::1] _pos
    cdef Py_ssize_t _num_orn, _num_grn

    def __init__(self, Layers.LiHopfield olf, Layers.BAM asso, Layers.Single gus, 
                    Movement.RadMotor m, Space.Space space, 
                    np.ndarray[np.int32_t] pfunc):
        """
        Parameters 
        ----------
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

        self._pfunc = pfunc

        self._preference = 0

        self._I1 = np.zeros(o)

        self._recalled_default = np.zeros(g, dtype=np.float64)
        self._recalled = self._recalled_default


    cpdef void set_target(self):
        pos = self._m.get_pos()
        target = self._space.near(pos)
        self._m.heading(target)


    cpdef void step(self):
        self._pos[0] = self._m.get_pos()

        self._stim = self._space.stim_at(<Py_ssize_t> self._pos[0, 0], 
                                         <Py_ssize_t> self._pos[0, 1])

        self._I1 = self._olf.feed(self._stim[:self._num_orn])
        cdef double[::1] I2

        cdef bint allzero = True
        for i in range(self._num_grn):
            if self._stim[i+self._num_orn] != 0:
                allzero = False
                break

        cdef double *to_judge
        self._preference = 0

        if allzero: # all are zero
            self._recalled = self._asso.recall(self._I1)
            to_judge = &self._recalled[0]
        else:
            I2 = self._gus.feed(self._stim[self._num_orn:])
            self._asso.learn(self._I1, I2)

            self._recalled = self._recalled_default

            to_judge = &self._stim[self._num_orn]
        

        judge(&self._preference, to_judge, &self._pfunc[0], self._num_grn)
        
        self._m.set_preference(self._preference)

        self._pos[0] = self._m.move()


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