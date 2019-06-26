#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
import numpy as np
from Stimuli import *
from Layers import *
from Environment import *

def test_Stimuli():
    num_s = 10000
    rang = 1000

    fail = []

    att = lambda x: x * 2

    s = Stimuli(num_s, 5, 5, att)
    pos = s.get_pos()
    kd = LazyKDTree(s)

    stims = '\n###\nStimuli:\n'
    for i in range(num_s):
        p = pos[i]
        matched_s, dist = kd.near(p)
        if not (matched_s == p).all():
#            # Comment out to show details
#            print('Case #', i)
#            print('Not the nearest neighbor at stimulus ', p)
#            print('\tMatched stimulus: ', matched_s)
            fail.append(dist)

        stims += str(p) + ('\n' if (i + 1) % 8 == 0 else '\t')

    print('\nNumber of stimuli: ', num_s)
    print('Stimuli Density: {:.3f} per unit square'.format(num_s / rang**2))
    print('Smallest unit: 1')

    if fail:
        print('\nFail Case Number: ', len(fail))
        print('Fail Case* Percentage: {:.4f}'.format(len(fail) / num_s))
        print('Average Fail Case Distance Offset: ', np.average(fail))
        print('Fail Case Distance Offset STD: ', np.std(fail))
        print('*Fail case: fail to find the exact nearest neighbor')

#   # Command-line UI for testing
#    while True:
#        print(stims)
#
#        pos = input("pos (split by a single comma ',') >> ").split(',')
#        pos = np.array([float(pos[0]), float(pos[1])])
#
#        s, d = kd.near(pos, _eap=True)
#        print('Result:')
#        print(s.pos())
#        print(d)

def test_Env():
    num_s = 1000
    epoch = 2000
    max_pos = 500
    gus_T = 3
    orn = 10
    grn = 5

    olf = LiHopfield(orn, name='Olf')
    gus = Single(grn, name='Gus')

    mapping = lambda x: np.array([x[i] + x[i+1] for i in range(0, orn, 2)])

    stims = Stimuli(num_s, orn, grn, mapping, max_pos, gus_T)

    print('Number of stimuli: ', num_s)
    print('Max x or y value: ', max_pos)
    print('ORN size: ', orn)
    print('GRN size:  {}\tGustatory Threshold:  {}'.format(grn, gus_T))
    print('Number of epochs: ', epoch)

    print('\nEstablishing the environment...')
    e = Environment(olf, gus, stims)
    print('Start simulation...')
    e.sim(epoch=epoch, _eap=True)


if __name__ == '__main__':
    print('Testing Stimuli.Stimuli and Stimuli.LazyKDTree...')
    test_Stimuli()

    print('\n'+ '#' * 80)
    print('Testing Environment.Environment...')
    test_Env()