import numpy as np
from numpy.linalg import inv
import pandas as pd

"""
Black-Litterman model

We have two views that we would like to incorporate into the model.
First, we hold a strong view that the Money Market rate will be 2% next year.
Second, we also hold the view that S&P 500 will outperform 10-year Treasury Bonds by 5%
but we are not as confident about this view.
"""
#MoneyMarket mean = 0.02         strong view: 0.00001
#Stocks - Bonds = 0.05           weaker view: 0.001
tau = 0.1

P = np.array([[0,0,1],[1,-1,0]])     
q = np.array([[0.02],[0.05]])
omega = np.array([[0.00001,0],[0,0.001]])
sigma = covM(rates).values.astype(float)

def blm_mu(P,q,omega,tau,sigma):
    blmdf = pd.Series(index = df.columns)
    first = inv(inv(tau*sigma) + np.dot(np.dot(np.transpose(P),inv(omega)),P))
    second = np.dot(inv(tau*sigma),geodf) + np.transpose(np.dot(np.dot(np.transpose(P),inv(omega)),q))
    mu = np.dot(first, np.transpose(second))
    for i in range(len(mu)):
        blmdf.iloc[i] = mu[i]
    return blmdf
blmdf = blm_mu(P,q,omega,tau,sigma)

b = list(df.columns.values)
b.insert(0,'Variance')

def bdf(Rs, Re, incr, mean):
    ER_BLM = np.arange((Rs/100), ((Re+0.1)/100), incr/100)
    BLM_df = pd.DataFrame(columns = b)
    for ers_blm in ER_BLM:
        BLM_df.loc[ers_blm] = MVO(ers_blm,mean).loc[ers_blm]
    return BLM_df

#Example
#Solving for target value R = 4.0% to R = 11.5% with increments of 0.5%
#we now get the optimal portfolios
bdf(4.0, 11.5, 0.5, blmdf)    
#          Variance    Stocks     Bonds            MM
#   0.040  0.001240  0.081951  0.171298  7.467506e-01
#   0.045  0.001539  0.115402  0.207045  6.775521e-01
#   0.050  0.001957  0.148854  0.242793  6.083530e-01
#   0.055  0.002496  0.182305  0.278541  5.391538e-01
#   0.060  0.003155  0.215757  0.314289  4.699547e-01
#   0.065  0.003934  0.249208  0.350036  4.007558e-01
#   0.070  0.004834  0.282659  0.385784  3.315566e-01
#   0.075  0.005853  0.316111  0.421532  2.623576e-01
#   0.080  0.006993  0.349562  0.457279  1.931585e-01
#   0.085  0.008252  0.383014  0.493027  1.239595e-01
#   0.090  0.009632  0.416465  0.528774  5.476062e-02
#   0.095  0.011148  0.467381  0.532618  4.028017e-07
#   0.100  0.013290  0.584535  0.415465  1.905159e-07
#   0.105  0.016287  0.701688  0.298312  9.674401e-08
#   0.110  0.020140  0.818842  0.181158  1.201214e-07
#   0.115  0.024848  0.935995  0.064005  4.601774e-08
