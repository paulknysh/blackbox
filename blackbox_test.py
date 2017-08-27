#!/usr/bin/env python
# coding=utf8

import blackbox as bb
import numpy as np


def fun(par):
    """
    toy optimisation test problem
    """
    a, b = par
    return np.sqrt((a - 5) ** 2 + (b - 3) ** 2)


def test_blackbox():

    data = bb.search(f=fun, box=[[-10., 10.], [-10., 10.]],
                     n=10, m=10, batch=8)
    print(data[0])


if __name__ == '__main__':
    test_blackbox()
