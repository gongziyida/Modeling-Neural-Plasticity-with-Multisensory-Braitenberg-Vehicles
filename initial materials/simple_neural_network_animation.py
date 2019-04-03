#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#DATE: Wed Apr  3 16:15:13 2019
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from neural_network import neural_network

stimuli = np.random.rand(6, 11)
c = neural_network(stimuli=stimuli, threshold=0.05)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches((12, 5))

ax1.scatter(stimuli[:, -2], stimuli[:, -1], s=10, c='r')

agent = ax1.scatter(0, 0, s=10, c='b')
ax1.legend(labels=['Stimuli', 'Agent'])

xlabels = ['Odor\n1', 'Odor\n2', 'Odor\n3', 'Odor\n4', 'Odor\n5',
           'Odor\n6', 'Taste\n1', 'Taste\n2', 'Taste\n3']
sensors = ax2.bar(xlabels, c.get_sensory_inputs())
ax2.set_title('Agent\'s sensory inputs')

def update(pos):
    agent.set_offsets(pos)
    sensory_inputs = c.jump(pos)
    max_h = sensory_inputs.max() + 1
    ax2.set_ylim([0, max_h])

    for bar, h in zip(sensors, sensory_inputs):
        bar.set_height(h)
    return agent, sensors

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, interval=1, repeat_delay=10,
                         frames=[(i/100, i/100) for i in range(1, 100)])
    anim.save('animation.gif', dpi=100, writer='pillow')
    plt.show()
