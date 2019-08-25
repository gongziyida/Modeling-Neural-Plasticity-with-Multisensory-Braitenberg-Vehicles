""" Thanks @nathancy on StackOverflow
    Website: 
    stackoverflow.com/questions/56486710/animate-pyqtgraph-in-class?rq=1
"""
from pyqtgraph.Qt import QtCore, QtGui
from threading import Thread
import pyqtgraph as pg
import numpy as np
import random
import sys
import time
import os

import Simulation
import Layers
import Movement
import Space

class Environment(QtGui.QWidget):
    def __init__(self, sim, freq=0.01, change_space_at_step=10000):
        super().__init__()

        # the simulation module
        self.sim = sim

        # Desired Frequency (Hz) = 1 / self.FREQUENCY
        # USE FOR TIME.SLEEP (s)
        self.FREQUENCY = freq

        # Frequency to update plot (ms)
        # USE FOR TIMER.TIMER (ms)
        self.TIMER_FREQUENCY = self.FREQUENCY * 1000

        # Outer grid layout
        self.layout = QtGui.QGridLayout()
        
        self._make_env()
        self._make_data_plots()

        # Count the steps of the BV
        self.step_counter = 0

        # when to change the space
        self.time_to_change = change_space_at_step

        # start the simulation
        self.read_pos_thread()
        self.start()


    def _make_env(self):
        # Create Plot Widget 
        self.env_widget = pg.PlotWidget()

        # Enable/disable plot squeeze (Fixed axis movement)
        # self.env_widget.plotItem.setMouseEnabled(x=False, y=False)
        max_pos = self.sim.get_max_pos()
        self.env_widget.setXRange(0, max_pos)
        self.env_widget.setYRange(0, max_pos)
        self.env_widget.setTitle('Environment')

        # Making moving BV
        self.moving_BV = pg.ScatterPlotItem()
        self.moving_BV.setBrush(255,0,0)
        # Add moving BV to the plot widget
        self.env_widget.addItem(self.moving_BV)

        # Making static stimuli
        self.stims = pg.ScatterPlotItem()
        self.stims.setData(pos=self.sim.get_stim())
        # Add stimuli to the plot widget
        self.env_widget.addItem(self.stims)

        # Add to the outer layoit
        self.layout.addWidget(self.env_widget, 0, 0, 2, 2)

        # set the scale of the first column (2 unit column width)
        self.layout.setColumnStretch(0, 2)


    def _make_data_plots(self):
        # the preference data
        self.pref_data = np.zeros(50, dtype=np.float64)

        # Making preference widget and plot
        self.pref_widget = pg.PlotWidget()
        self.pref_widget.setYRange(-100, 100)
        self.pref_widget.setTitle('Preference')
        self.pref_widget.addLegend(offset=(0, 0))

        self.pref_plot = pg.PlotCurveItem(name='actual')
        self.pref_widget.addItem(self.pref_plot)

        self.ideal_pref_plot = pg.PlotCurveItem(name='ideal')
        self.ideal_pref_plot.setPen(255, 255, 0)
        self.pref_widget.addItem(self.ideal_pref_plot)

        self.layout.addWidget(self.pref_widget, 0, 2, 1, 1)
        
        # the stimulus received
        o, g = self.sim.get_num_receptors()
        self.stim = np.zeros(o + g)
        self.olf_processed = np.zeros(o)
        self.recalled = np.zeros(g)
        self.gus_x_axis = np.arange(o, o+g)

        # Making stimulus widget and plot
        xticks = [(i, 'O{}'.format(i)) for i in range(o)] + \
                [(i+o, 'G{}'.format(i)) for i in range(g)]
        ax = pg.AxisItem(orientation='bottom')
        ax.setTicks([xticks])
        self.stim_widget = pg.PlotWidget(axisItems={'bottom': ax})
        self.stim_widget.setYRange(0, 3)
        self.stim_widget.setTitle('Stimulus')
        self.stim_widget.addLegend(offset=(0, 10))

        # received taste
        self.stim_plot = pg.PlotCurveItem(name='received')
        self.stim_widget.addItem(self.stim_plot)
        
        # recalled taste
        self.recalled_plot = pg.PlotCurveItem(name='recalled')
        self.recalled_plot.setPen(255, 0, 0)
        self.stim_widget.addItem(self.recalled_plot)

        # ideal taste
        self.ideal_gus_plot = pg.PlotCurveItem(name='ideal')
        self.ideal_gus_plot.setPen(255, 255, 0)
        self.stim_widget.addItem(self.ideal_gus_plot)

        self.layout.addWidget(self.stim_widget, 1, 2, 1, 1)

        # set the scale of the 3rd column
        self.layout.setColumnStretch(2, 1)

        # the processed olfactory information
        self.olf_processed_widget = pg.PlotWidget()
        self.olf_processed_widget.setTitle('Output of Olfactory Bulb')
        self.olf_processed_widget.setYRange(0, 1)
        self.olf_processed_plot = pg.PlotCurveItem()
        self.olf_processed_widget.addItem(self.olf_processed_plot)
        self.layout.addWidget(self.olf_processed_widget, 0, 3, 1, 1)

        # the associative memory weight matrix
        self.weight_widget = pg.PlotWidget()
        self.weight_widget.setTitle('Associative Memory Weight Matrix')
        self.weight_heat = pg.ImageItem()
        self.weight_widget.addItem(self.weight_heat)
        self.layout.addWidget(self.weight_widget, 1, 3, 1, 1)
        self.W = np.asarray(self.sim.get_asso_weight())

        # set the scale of the 4th column
        self.layout.setColumnStretch(3, 1)



    # Update plot
    def start(self):
        self.pos_update_timer = QtCore.QTimer()
        self.pos_update_timer.timeout.connect(self.plot_updater)
        self.pos_update_timer.start(self.TIMER_FREQUENCY)

  
    # Read in data using a thread
    def read_pos_thread(self):
        self.BV_pos = self.sim.get_BV_pos()
        self.pos_update_thread = Thread(target=self.read_pos, args=())
        self.pos_update_thread.daemon = True
        self.pos_update_thread.start()


    def read_pos(self):
        frequency = self.FREQUENCY
        while True:
            if self.step_counter == self.time_to_change:
                self.sim.change_space()
                self.step_counter = 0
                self.stims.setData(pos=self.sim.get_stim())


            if self.step_counter % 1000 == 0:
                self.sim.set_target()
            self.sim.step()
            self.step_counter += 1
            

            # update BV pos
            self.BV_pos = self.sim.get_BV_pos()

            # update pref
            self.pref_data = np.asarray(self.sim.get_pref())

            # update ideal pref
            self.ideal_pref_data = np.asarray(self.sim.get_ideal_pref())

            # update stimulus
            self.stim = np.asarray(self.sim.get_cur_stim())
            
            # update processed olfactory information
            self.olf_processed = np.asarray(self.sim.get_olf_bulb_output())
            
            # update recalled taste
            self.recalled = np.asarray(self.sim.get_recalled())

            # update ideal taste
            self.ideal_gus = np.asarray(self.sim.get_ideal_gus())
            if np.isnan(self.ideal_gus).any():
                print(np.asarray(self.BV_pos))

            # update weight
            self.W = np.asarray(self.sim.get_asso_weight())

            # sleep
            time.sleep(frequency)


    def plot_updater(self):
        self.moving_BV.setData(pos=self.BV_pos)
        self.pref_plot.setData(y=self.pref_data)
        self.ideal_pref_plot.setData(y=self.ideal_pref_data)
        self.stim_plot.setData(y=self.stim)
        self.olf_processed_plot.setData(y=self.olf_processed)
        self.recalled_plot.setData(x=self.gus_x_axis, y=self.recalled)
        self.ideal_gus_plot.setData(x=self.gus_x_axis, y=self.ideal_gus)
        self.weight_heat.setImage(self.W)


    def get_layout(self):
        return self.layout

    def get_BV_pos(self):
        return self.BV_pos

    def get_env_widget(self):
        return self.env_widget



if __name__ == '__main__':
    o = 10
    g = 5

    def mapping(x, y): 
        for i in range(g):
            y[i] = 0

        for i in range(o):
            y[i // 2] += x[i]

        norm = np.linalg.norm(y)

        if norm > 1e-15:
            for i in range(g):
                y[i] /= norm

    space = Space.Space(20, o, g, mapping)                                 

    gus = Layers.Single(g, lambda x: x)                                     
    
    olf = Layers.LiHopfield(o, tau=5)                                             
    
    m = Movement.RadMotor(200)                                              
    
    asso = Layers.BAM(o, g, adapting_rate=1e-3, depression_rate=1e-10, enable_dep=True) 
    
    pfunc = np.array([0, 0, 1, 1], dtype=np.int32)

    sim = Simulation.OnlineSim(olf, asso, gus, m, space, pfunc, mapping)
    
    # save imgs
    if not os.path.exists('img'):
        os.makedirs('img')

    space.save_img(name_prefices=('img/odor_space', 'img/taste_space'))
    olf.save_img(name='img/olf.png')
    asso.save_img(name='img/asso.png')
    sim.save_network_outer_img(name='img/outer_struct.png')


    # Create main application window
    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create("Cleanlooks"))
    mw = QtGui.QMainWindow()
    mw.setWindowTitle('Plot Example')

    # Create scrolling plot
    env_widget = Environment(sim)

    # Create and set widget layout
    # Main widget container
    cw = QtGui.QWidget()
    ml = QtGui.QGridLayout()
    cw.setLayout(ml)
    mw.setCentralWidget(cw)

    # Can use either to add plot to main layout
    #ml.addWidget(env_widget.get_env_widget(),0,0)
    ml.addLayout(env_widget.get_layout(),0,0)
    mw.show()

    # Start Qt event loop unless running in interactive mode or using pyside
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()