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
w_equ = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])
my_del = 2.5


def mycov(cor, sd):
    cov =np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j]= cor[i, j]*(sd[i]*sd[j])
    return cov


cov = mycov(cor, sd)
mean_equ = my_del*np.dot(cov, w_equ)

cov_inv = np.linalg.inv(cov)
w_opt = np.dot(cov_inv, mean_equ)/my_del

# 1A grpha
mu_1A = np.repeat(0.07, 7)
w_1A = np.dot(cov_inv, mu_1A)/my_del
mu_1A_shift = np.array([0.07, 0.07, 0.045, 0.095, 0.07, 0.045, 0.07])
w_1A_shift = np.dot(cov_inv, mu_1A_shift)/my_del

# 1B graph
eu_sum = w_equ[2]*mean_equ[2] + w_equ[3]*mean_equ[3] + w_equ[5]*mean_equ[5]
eu_m = np.array([[w_equ[2]/(w_equ[2] + w_equ[5]), w_equ[5]/(w_equ[2] + w_equ[5]), -1],
                  [w_equ[2], w_equ[5], w_equ[3]],
                  [1, -1, 0]])
b1 = np.array([[-0.05], [eu_sum], [0.016]])
mu_1B = np.dot(np.linalg.inv(eu_m), b1)

# 1C graph
mu_1C = np.array([mean_equ[0], mean_equ[1], mu_1B[0], mu_1B[2], mean_equ[4], mu_1B[1], mean_equ[6]])
w_1C = np.dot(cov_inv, mu_1C)/my_del


