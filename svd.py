#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np

def svd(x):
    x = np.matrix(x)
    m,n = x.shape
    if m < n:
        x = x.T
    s,u = np.linalg.eig(x * x.T)
    u = u[np.argsort(s)[::-1]]
    s,v = np.linalg.eig(x.T * x)
    s = np.sqrt(s)
    v = np.transpose(v)
    sort = np.argsort(s)[::-1]
    s = s[sort]
    v = v[sort]
    if m < n:
        u_T = np.transpose(u)
        return v,s,u_T
    v_T = np.transpose(v)
    return u,s,v_T


if __name__ == '__main__':
    x = np.arange(1,9).reshape((4,2))
    print("my svd:")
    u,s,v = svd(x)
    print("u:\n", u)
    print("s:\n", s)
    print("v:\n", v)
    print("numpy svd:")
    u,s,v = np.linalg.svd(x)
    print("u:\n", u)
    print("s:\n", s)
    print("v:\n", v)
