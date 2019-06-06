from pyomo.environ import (ConcreteModel, Var, Objective, Constraint,
                           NonNegativeReals, minimize, value)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from fractions import Fraction
import numpy as np
import pandas as pd
import pprint

#mean_equ = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
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
    cov =np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j]= cor[i, j]*(sd[i]*sd[j])
    return cov


cov = mycov(cor, sd)
my_del = 2.5
cov_inv = np.linalg.inv(cov)
mean_equ = my_del*np.dot(cov, w_equ)
mean_equ.reshape((7,1))
w_opt = np.dot(cov_inv, mean_equ)/my_del

# Chart 2A
P = np.array([[0.0, 0.0, -w_equ[2]/(w_equ[2] + w_equ[5]), 
              1.0, 0.0, -w_equ[5]/(w_equ[2] + w_equ[5]), 0]])
PT = P.transpose()
Q = np.array([[0.05]])
my_scalar = np.dot(P, np.dot(cov, PT))
t1 = np.linalg.inv(cov_inv + np.dot(PT, P)/my_scalar)
t2 = np.dot(cov_inv, mean_equ) + Q*PT/my_scalar
mu_BL = np.dot(t1, t2)

# Chart 2B
w_2B = np.dot(cov_inv, mu_BL)/my_del

# Chart 2C
w_dev = -w_equ + w_2B
w_2C = np.round(w_dev, 2)