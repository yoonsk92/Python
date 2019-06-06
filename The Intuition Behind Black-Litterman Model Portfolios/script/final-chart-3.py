from pyomo.environ import (ConcreteModel, Var, Objective, Constraint,
                           NonNegativeReals, minimize, value)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from fractions import Fraction
import numpy as np
import pandas as pd
import pprint

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


# Define function for updating Black-Litterman mean
def myML(P, Q):
    tau = 0.05
    Ome = tau*np.dot(P, np.dot(cov, P.transpose()))
    Ome = np.diagonal(Ome)
    Ome = np.diag(Ome)
    Ome_inv = np.linalg.inv(Ome)
    t1 = np.linalg.inv(tau*cov)
    t2 = np.dot(P.transpose(), np.dot(Ome_inv, P))
    t3 = np.dot(np.linalg.inv(tau*cov), mean_equ)
    t4 = np.dot(np.dot(P.transpose(), Ome_inv), Q)
    mean_BL = np.dot(np.linalg.inv(t1 + t2), (t3+t4))
    return(mean_BL)

# Chart 3A
P = np.array([[0, 0, -w_equ[2]/(w_equ[2] + w_equ[5]), 1, 0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0],
               [0, 1, 0, 0, 0, 0, -1]])
Q = np.array([[0.05], [0.03]])

mu_BL = myML(P, Q)
w_BL = np.dot(cov_inv, mu_BL)/my_del 
w_dev = w_BL - w_equ 
w_3AG = np.round(w_dev, 4)*100
w_3AR = 100*P[0]
w_3AB = 100*P[1]


df1 = pd.DataFrame(w_3AR , columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
df2 = pd.DataFrame(w_3AB , columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
df3 = pd.DataFrame(w_3AG , columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
frames = [df1,df2,df3]
result = pd.concat(frames, keys = ['Germany VS Europe', 'Canada VS US','Optimal Deviations']).reset_index()
ax = sns.barplot(x="level_1", y="weight", hue="level_0", data=result)

##Chart 3B
w_3B = 100*w_BL
df1 = pd.DataFrame(100*w_equ, columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
df2 = pd.DataFrame(w_3B  , columns=['weight'],index = ['AUS','CAD','FR','GER','JAP','UK','US'])
frames = [df1,df2]
result = pd.concat(frames, keys = ['Equilibrium', 'Optimal']).reset_index()
ax = sns.barplot(x="level_1", y="weight", hue="level_0", data=result)

