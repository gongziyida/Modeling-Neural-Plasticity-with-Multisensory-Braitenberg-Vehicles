#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#AUTHOR: Ziyi Gong
#VERSION: 0.1
#PYTHON_VERSION: 3.6
'''
DESCRIPTION

'''
def arg_check(arg, name, types):
    if arg is None:
        return

    if types == 'callable':
        if not callable(arg):
            raise TypeError("Argument '{}' must be callable.".format(name))
    elif not isinstance(arg, types):
        raise TypeError("Argument '{}' must be {}.".format(name, types))


def collection(c, name, types, size=None):
    if not (size is None):
        if len(c) != int(size):
            raise TypeError("Collection '{}' must be of size {}."\
                            .format(name, size))

    for i in c:
        if types == 'callable':
            if not callable(i):
                raise TypeError("Collection '{}' must all be callable."\
                                .format(name))
        else:
            if not isinstance(i, types):
                raise TypeError("Collection '{}' must all be {}." \
                                .format(name, types))

def is_within(a, limits):
    if a > limits[1] or a < limits[0]:
        raise ValueError('Value {} is out of bounds (limit: {}).'\
                        .format(a, limits))