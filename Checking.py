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
        if callable(arg):
            return
        else:
            raise TypeError("Argument '{}' must be callable.".format(name))

    if isinstance(arg, types):
        return
    else:
        raise TypeError("Argument '{}' must be {}.".format(name, types))


def collection(c, name, types):
    for i in c:
        if types == 'callable':
            if not callable(i):
                raise TypeError("Collection '{}' must all be callable."\
                                .format(name))

        else:
            if not isinstance(i, types):
                raise TypeError("Collection '{}' must all be {}." \
                                .format(name, types))