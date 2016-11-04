#!/usr/bin/env python
"""Example from arXiv note (https://arxiv.org/pdf/1605.00998.pdf)"""
import blackbox as bb


def fun(par):
    x = par[0]
    y = par[1]
    return abs(x-y) + ((x+y-1)/3)**2


def main():
    bb.search(f=fun,
              box=[[-1.,1.],[-1.,1.]],
              n=8,
              it=8,
              cores=2,
              resfile='output.csv')


if __name__ == '__main__':
    main()
