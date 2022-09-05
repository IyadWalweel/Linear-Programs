import numpy as np
from Stopping_Criteria.KKT import KKT
from Stopping_Criteria.LP_def import check_None, DFun
from Stopping_Criteria.DGKKT import DGKKT
from Stopping_Criteria.SDG import SDG

def crit_Test(stop_crit, fstar, xstar, dual, V, c, X, Y = None, Al = None, bl = None, Ag = None, bg = None, 
                                                                            Ae = None, be = None, ite = 0, tol = 1e-6):
    F, D = [], []
    if (stop_crit == 'FNOI'):
        for i in range(ite):
            F.append(V[i] - fstar)
            D. append(np.linalg.norm(X[i] - xstar))
    elif (stop_crit == 'ASD'):
        i = 0
        x_prev = X[0]
        x_curr = X[1]
        F.append(V[0] - fstar)
        D.append(np.linalg.norm(x_prev - xstar))
        while (np.linalg.norm(x_curr - x_prev) >= tol):
            i += 1 
            x_prev = x_curr
            x_curr = X[i+1]
            F.append(V[i] - fstar)
            D.append(np.linalg.norm(x_prev - xstar))
    elif(stop_crit == 'KKT'):
        F.append(V[0] - fstar)
        D.append(np.linalg.norm(X[0] - xstar))
        i = 0
        while (KKT(c, X[i], Y[i], Al, bl, Ag, bg, Ae, be, tol) >= tol):
            i+=1
            F.append(V[i] - fstar)
            D.append(np.linalg.norm(X[i] - xstar))
    elif(stop_crit == 'DG'):
        i = 0
        Al, bl, Ag, bg, Ae, be = check_None(c, Al, bl, Ag, bg, Ae, be)
        mu, nu, y = dual[0], dual[1], dual[2]
        dp = DFun(bl, bg, be, mu, nu, y)
        F.append(V[i] - fstar)
        D.append(np.linalg.norm(X[i] - xstar))
        while (V[i] - dp >= tol):
            i += 1
            F.append(V[i] - fstar)
            D.append(np.linalg.norm(X[i] - xstar))
    elif(stop_crit == 'DGKKT'):
        i = 0
        RhoP, RhoD, RhoA = DGKKT(c, X[0], Y[0], Al, bl, Ag, bg, Ae, be)
        dp = max(1, RhoP)
        dd = max(1, RhoD)
        while(RhoP >= tol or RhoD >= tol or RhoA >= tol):
            F.append(V[i] - fstar)
            D.append(np.linalg.norm(X[i] - xstar))
            RhoP, RhoD, RhoA = DGKKT(c, X[i], Y[i], Al, bl, Ag, bg, Ae, be, dp=dp, dd=dd)
            i += 1
    elif(stop_crit == 'SDG'):
        i = 0
        F.append(V[i] - fstar)
        D.append(np.linalg.norm(X[i] - xstar))
        while (SDG(X[i], Y[i], X[i], Y[i], c, Al, bl, Ag, bg, Ae, be) >= tol):
            i += 1
            F.append(V[i] - fstar)
            D.append(np.linalg.norm(X[i] - xstar))

    return F, D


    