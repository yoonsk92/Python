import pandas as pd
import numpy as np

correlation = np.array([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
                        [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
                        [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
                        [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
                        [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
                        [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
                        [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])

vol = np.array([0.16, 0.203, 0.248, 0.271, 0.21, 0.20, 0.187])
wgt = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])
rt = np.array([0.039, 0.069, 0.084, 0.09, 0.043, 0.068, 0.076])

N,M = correlation.shape

def cov(cor,vol):
    cov = np.zeros((7,7))
    for i in range(M):
        for j in range(N):
            cov[i,j] = cor[i,j]*(vol[i]*vol[j])
    return cov

cov_vol = cov(correlation, vol)
cov_wgt = cov(correlation, wgt)
cov_rt = cov(correlation, rt)

def risk_constrained(EQ,sigma,vol,risk_aversion):
    w_star = np.dot(np.linalg.inv(np.dot(risk_aversion,sigma)),EQ)
    w_r = np.dot(vol,w_star) / np.sqrt(np.dot(np.dot(w_star.transpose(),sigma),w_star))
    return w_r
























    












