from pyomo.environ import (ConcreteModel, Var, Objective, Constraint,
                           Reals, minimize, maximize, value)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from fractions import Fraction
import numpy as np
import pandas as pd
import pprint
import math
import seaborn as sns

# mean_equ = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
cor = np.array([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
                [0.488, 1.000, 0.663, 0.655, 0.310, 0.608, 0.779],
                [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
                [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
                [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
                [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
                [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
sd = np.array([0.16, 0.203, 0.248, 0.271, 0.21, 0.20, 0.187])
#  seq: Australia, Canada, France, Germany, Japan, UK, USA
N = cor.shape[0]
w_equ = np.array([[0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615]])
w_equ = np.reshape(w_equ, (7, 1))


def mycov(cor, sd):
    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j] = cor[i, j]*(sd[i]*sd[j])
    return cov


cov = mycov(cor, sd)
my_del  = 2.5
cov_inv = np.linalg.inv(cov)
mean_equ = my_del*np.dot(cov, w_equ)
mean_equ.reshape((7, 1))
w_opt = np.dot(cov_inv, mean_equ)/my_del


# Define function for updating Black-Litterman mean. new variable
# conf is a N*N size diagonal matrix with confident level
def myML(P, Q, conf):
    tau = 0.05
    PS = np.dot(P, np.dot(cov, P.transpose()))
    Ome = np.diagonal(tau*PS)
    Ome = np.diag(Ome)
    Ome_adj = np.dot(Ome, conf)
    Ome_inv = np.linalg.inv(Ome_adj)
    t1 = np.linalg.inv(tau*cov)
    t2 = np.dot(P.transpose(), np.dot(Ome_inv, P))
    t3 = np.dot(np.linalg.inv(tau*cov), mean_equ)
    t4 = np.dot(np.dot(P.transpose(), Ome_inv), Q)
    mean_BL = np.dot(np.linalg.inv(t1 + t2), (t3 + t4))
    l1 = (tau/my_del)*np.dot(Ome_inv, Q)
    itm = np.linalg.inv(Ome_adj/tau + PS)
    l2 = np.dot(np.dot(np.dot(itm, P), cov), w_equ)
    l3 = np.dot(np.dot(itm, PS), l1)
    Lambda = l1 - l2 - l3
    return([mean_BL, Lambda])


# Chart 5
P = np.array([[0, 0, -w_equ[2]/(w_equ[2] + w_equ[5]), 1, 0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0],
              [0, 1, 0, 0, 0, 0, -1]])
Q = np.array([[0.05], [0.03]])
conf = np.identity(2)
mu_BL = (myML(P, Q, conf))[0]
w_port = (myML(P, Q, conf))[1]
wstar = np.dot(cov_inv, mu_BL)/my_del
wstar_vol = np.dot(np.dot((wstar.transpose()), cov), wstar)
w_cons = 0.2*wstar/math.sqrt(wstar_vol)
w_dev = w_cons - w_equ
# Prepare the data
mu_in = mu_BL.reshape(7,)
# MVO model
# Building model
def alloc(mean, co_var, vol, beta):
    nMean = mean.shape[0]
    nX = np.arange(nMean)
    model = ConcreteModel()
    model.x = Var(nX, within=Reals)

    model.obj = Objective(expr=
                          sum(mean[i]*model.x[i] for i in nX), sense=maximize)
    model.con1 = Constraint(expr=
                            (0, sum(co_var[i][j]*model.x[i]*model.x[j] for i in nX for j in nX), vol**2))
    model.con2 = Constraint(expr=(sum(model.x[t] for t in nX) == 1))
    model.con3 = Constraint(expr=
                            (sum(co_var[i][j]*model.x[i]*beta[j] for i in nX for j in nX))==
                            (sum(co_var[i][j]*beta[i]*beta[j] for i in nX for j in nX)))
    model.pprint()
    return model


# calculation
def MVO(mean, co_var, vol, beta):
    opt = SolverFactory("ipopt")
    results = []
    nMean = mean.shape[0]
    for u in np.arange(len(vol)):
        m = alloc(mean, co_var, vol[u], beta)
        r = opt.solve(m)
        print('Solver Status: ',  r.solver.status)
        print('Solver Terminate: ', r.solver.termination_condition)
        assert (r.solver.status == SolverStatus.ok) and (r.solver.termination_condition == TerminationCondition.optimal)
        _x = np.zeros(nMean)
        for v in np.arange(nMean):
            _x[v] = value(m.x[v])
        results.append({'return': value(m.obj), 'alloc': _x})
    return results

# Result
Vol = [0.2]
beta = w_equ.reshape((7, ))
results = MVO(mu_in, cov, Vol, beta)
pprint.pprint(results,width=300)
y_results_re = []
y_results_alloc = []
for k in range(len(results)):
    y_results_re.append(results[k]['return'])
for k in range(len(results)):
    y_results_alloc.append(results[k]['alloc'])
    
w_7 = (y_results_alloc[0]).reshape((7,1))
w_dev = w_7 - w_equ

def plot_weight(w_0,w_1,w_2):
    df1 = pd.DataFrame(w_0, columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
    df2 = pd.DataFrame(w_1, columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
    df3 = pd.DataFrame(w_2, columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
    frames = [df1,df2,df3]
    result = pd.concat(frames, keys = ['Equilibrium','Constrained', 'Deviation']).reset_index()
    ax = sns.barplot(x="level_1", y="weight", hue="level_0", data=result)
    return ax
