#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION:
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
from Stimuli import *
import numpy as np

def test_Stimuli():
    num_s = 10000
    rang = 1000
    att = np.array([1, 2, 3])

    fail = []

    s = [Stimulus(att, att, np.random.randint(rang, size=2)) for i in range(num_s)]
    kd = KDTree(s)

    s.sort(key=lambda s: s.pos()[0])


    stims = '\n###\nStimuli:\n'
    for i in range(num_s):
        pos = s[i].pos()
        matched_s, dist = kd.near(pos)
        if not (matched_s.pos() == pos).all():
            print('Not the nearest neighbor at stimulus ', pos)
            print('\tMatched stimulus: ', matched_s.pos())
            fail.append(dist)

        stims += str(pos) + ('\n' if (i + 1) % 8 == 0 else '\t')

    print('\nNumber of stimuli: ', num_s)
    print('Stimuli Density: {:.3f} per unit square'.format(num_s / rang**2))
    print('Smallest unit: 1')

    if fail:
        print('\nFail Case* Percentage: {:.4f}'.format(len(fail) / num_s))
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

if __name__ == '__main__':
    test_Stimuli()