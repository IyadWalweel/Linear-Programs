import numpy as np
import cvxpy as cp
from tabulate import tabulate

from Stopping_Criteria.LP_def import PFun, g
from Simplex.Simplex.Simplex_Solver import simplex as splx


def solve(c, Al = None, bl = None, Ag = None, bg = None, Ae = None, be = None, obj = 'Min', 
                                                                show = False, Big_M = False, M = 100, eps = 1e-8):
    dual = []
    B = (bl, bg, be)
    d = len(c)
    x = cp.Variable(d)
    if (obj == 'max'):
        objective = cp.Maximize(PFun(c, x))
    else:
        objective = cp.Minimize(PFun(c, x))
    
    constraints = [x >= 0]
    if bl is not None:
        constraints += [g(Al, bl, x) <= 0]
    if bg is not None: 
        constraints += [g(Ag, bg, x) >= 0]
    if be is not None:
        constraints += [g(Ae, be, x) == 0]

    prob = cp.Problem(objective, constraints)
    res = prob.solve(solver = cp.GLPK)

    for i in range(len(B)):
        j = 0
        if B[i] is not None:
            dual.append(constraints[j+1].dual_value)
            j += 1
        else:
            dual.append(np.zeros(len(c)))

    result, table, X, Y, V, ite = splx(c, Al, bl, Ag, bg, Ae, be, obj, show, Big_M, M, eps)

    return (res, x.value, dual), (result, table, X, Y, V, ite)

