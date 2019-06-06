from pyomo.environ import (ConcreteModel, Var, Objective, Constraint,
                           NonNegativeReals, minimize, value)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from fractions import Fraction
import numpy as np
import pandas as pd
import pprint
import math

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
my_del = 2.5
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


# Chart 4 - 1st col
P_1 = np.array([[0, 0, -w_equ[2]/(w_equ[2] + w_equ[5]), 1, 0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0],
              [0, 1, 0, 0, 0, 0, -1]])
Q_1 = np.array([[0.05], [0.03]])
conf_1 = np.identity(2)
mu_BL_1 = (myML(P_1, Q_1, conf_1))[0]
w_port_1 = (myML(P_1, Q_1, conf_1))[1]
w_41 = np.dot(cov_inv, mu_BL_1)/my_del

# Chart 4 - 2nd col
P_2 = np.array([[0, 0, -w_equ[2]/(w_equ[2] + w_equ[5]), 1, 0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0],
               [0, 1, 0, 0, 0, 0, -1]])
Q_2 = np.array([[0.05], [0.04]])
conf_2 = np.identity(2)
mu_BL_2 = (myML(P_2, Q_2, conf_2))[0]
w_port_2 = (myML(P_2, Q_2, conf_2))[1]
w_42 = np.dot(cov_inv, mu_BL_2)/my_del

# Chart 4 -3rd col
conf_3 = np.diag([2.0, 1.0])
P_3 = np.array([[0, 0, -w_equ[2]/(w_equ[2] + w_equ[5]), 1, 0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0],
                [0, 1, 0, 0, 0, 0, -1]])
Q_3 = np.array([[0.05], [0.04]])
mu_BL_3 = (myML(P_3, Q_3, conf_3))[0]
w_port_3 = (myML(P_3, Q_3, conf_3))[1]
w_43 = np.dot(cov_inv, mu_BL_3)/my_del

df_1 = pd.DataFrame(w_port_1,index =['Germany vs Europe','Canads vs US'],columns = ['weights'])
df_2 = pd.DataFrame(w_port_2,index =['Germany vs Europe','Canads vs US'],columns = ['weights'])
df_3 = pd.DataFrame(w_port_3,index =['Germany vs Europe','Canads vs US'],columns = ['weights'])
frames = [df_1, df_2, df_3]
result = pd.concat(frames, keys = ['Example_3', 'Bullish', 'Less Confident']).reset_index()
ax = sns.barplot(x="level_0", y="weights", hue="level_1", data=result)