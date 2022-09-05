from unittest import skip
import numpy as np
from Stopping_Criteria.LP_def import neg_Element
from Stopping_Criteria.SDG import SDG


def V(x, xstar, c):
    return c@(x - xstar)

def F(x, A, b):
    return np.linalg.norm(A@x - b)


def K(x, y, c, A, b, beta = 1):
    dc = np.linalg.norm(neg_Element(y@A + c))
    pc = np.linalg.norm(A@x - b)
    dg = c@x + b@y
    res = dc**2 + pc**2 + dg**2

    return res

def D(x, y, c, A, b):   
    if (np.linalg.norm(A@x - b) <= 1e-7) and ((y@A + c >= -1e-8).all()):
        return abs(c@x + b@y)
    else:
        return 1e8


def S(x, y, c, A, b, beta = 1):
    res = SDG(x, y, x, y, c, Ae = A, be = b)
    return res

def measures(X, Y, xstar, c, A, b, beta = 1):
    v = []
    f = []
    k = []
    s = []
    d = []
    ks = []
    sk = []
    kd = []
    sd = []
    ds = []
    vs = []
    vk = []
    fs = []
    l = len(Y)
    for i in range(l):
        n = np.linalg.norm(X[i]) + np.linalg.norm(Y[i])
        v.append(V(X[i], xstar, c))
        f.append(F(X[i], A, b))
        k.append(K(X[i], Y[i], c, A, b, beta))
        s.append(S(X[i], Y[i], c, A, b, beta))
        d.append(D(X[i], Y[i], c, A, b))
        ks.append(2*beta*s[-1] + (2*s[-1] + n*np.sqrt(2*beta*s[-1]))**2)
        sk.append((1 + n)*np.sqrt(k[-1]) + (1/2*beta)*k[-1])
        kd.append(d[-1]**2)
        sd.append((1 + n)*d[-1] + (1/2*beta)*d[-1]**2)
        # print("SD = ", sd)
        ds.append(2*s[-1] + np.linalg.norm(X[i])*np.sqrt(2*beta*s[-1]))
        vs.append(2*s[-1] + n*np.sqrt(2*beta*s[-1]))
        vk.append(np.sqrt(k[-1]))
        fs.append(np.sqrt(2*beta*s[-1]))


    
    return v, f, k, s, d, ks, sk, kd, sd, ds, vs, vk, fs