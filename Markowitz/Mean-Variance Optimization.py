#Import modules
import numpy as np
import pandas as pd
from scipy import stats

#Import data
df = pd.read_csv("Sample data.csv").set_index('Year')

#Construct rates of returns table (RRT)
def rreturn(colname):
    rates = pd.DataFrame()
    index = df.columns.get_loc(colname)
    for row in range(len(df)-1):
        rates.loc[row,colname] = ((df.iloc[row+1,index] - df.iloc[row,index]) / (df.iloc[row,index]))*100
    return rates   
rates = pd.concat([rreturn('Stocks'),rreturn('Bonds'),rreturn('MM')], axis = 1)

#Calculate Arithmetic / Geometric Means of RRT
def mean_method(x, colname):
    if x == 'arith':
        a = (100 + rreturn(colname)) / 100
        arith_mean = (np.mean(a) - 1) * 100
        return arith_mean
    elif x == 'geo':
        g = (100 + rreturn(colname)) / 100
        geo_mean = pd.Series((stats.gmean(g)-1) * 100)
        return geo_mean
arthdf = pd.concat([mean_method('arith','Stocks'), mean_method('arith','Bonds'), mean_method('arith','MM')]) / 100
geodf = pd.concat([mean_method('geo','Stocks'), mean_method('geo','Bonds'), mean_method('geo','MM')]) / 100
geodf.index = ['Stocks','Bonds','MM']

#Calculate covariance of RRT
def cov(x1,x2):
    a = np.sum((rates[x1]-arthdf[x1])*(rates[x2]-arthdf[x2]))*(1/len(rates))*1/10000
    return a

#Construct Covariance Matrix of RRT
def covM(df):
    temp = df.columns.values
    b = pd.DataFrame(columns = temp, index = range(len(df.columns)))
    for col in range(len(df.columns)):
        for row in range(len(df.columns)):
            b.iloc[row,col] = cov(df.columns[row], df.columns[col])
    return b

#Markowitz' Mean-Variance Optimization
from pyomo.environ import *
def MVO(R,mean):        
    covmat = pd.DataFrame(data = covM(rates).values.astype(float), columns = df.columns, index = df.columns)
    mask = covmat.where(np.triu(np.ones(covmat.shape)).astype(np.bool))
    covmat_ = mask.stack().reset_index()
    covmat_.columns = ['Row', 'Column', 'Covariance']
    
    for i in range(len(covmat_)):
        if (covmat_.iloc[i,0] != covmat_.iloc[i,1]):
            covmat_.iloc[i,2] = covmat_.iloc[i,2] * 2
                
    model = ConcreteModel()
    X = list(df.columns.values)
    I = list(range(len(covmat_)))
    model.x = Var(X,within=NonNegativeReals)
    Row = list(covmat_.Row)
    Col = list(covmat_.Column)
    Covar = list(covmat_.Covariance)
    
    op = SolverFactory('ipopt')
    results = pd.DataFrame(columns = df.columns)

    model.obj = Objective(expr = sum(Covar[i]*model.x[row]*model.x[col] 
                            for i,row,col in zip(I, Row, Col)), sense = minimize)
    model.con1 = Constraint(expr = sum(mean.loc[j]*model.x[j] for j in X) >= R)
    model.con2 = Constraint(expr = sum(model.x[j] for j in X) == 1)
    
    op.solve(model)
    results.loc[R] = [model.x['Stocks'].value,
                        model.x['Bonds'].value,
                        model.x['MM'].value]
    return results

def rdf(Rs,Re, incr, mean):
    ER = np.arange((Rs/100), ((Re+0.1)/100), incr/100)
    rates_df = pd.DataFrame(columns = df.columns)
    for ers in ER:
        rates_df.loc[ers] = MVO(ers,mean).loc[ers]
    return rates_df

#Example using Sample Data.csv
#Solving Markowitz's MVO model for constructing a portfolio of
#US stocks, bonds, and cash using geometric / arithmetic means respectively.
#Target value R from 6.5% to 12% with increments of 0.5%.
rdf(6.5, 10.5, 0.5, geodf)
#            Stocks     Bonds            MM
#   0.065  0.027959  0.095843  8.761982e-01
#   0.070  0.135545  0.114035  7.504194e-01
#   0.075  0.243132  0.132227  6.246414e-01
#   0.080  0.350718  0.150419  4.988629e-01
#   0.085  0.458304  0.168613  3.730833e-01
#   0.090  0.565891  0.186804  2.473054e-01
#   0.095  0.673477  0.204995  1.215279e-01
#   0.100  0.782451  0.217543  6.159742e-06
#   0.105  0.931027  0.068973  4.538123e-07
    
rdf(6.5, 10.5, 0.5, arthdf)
#            Stocks     Bonds        MM
#   0.065  0.004788  0.097902  0.897310
#   0.070  0.086984  0.116683  0.796333
#   0.075  0.169182  0.135460  0.695359
#   0.080  0.251378  0.154239  0.594382
#   0.085  0.333576  0.173017  0.493407
#   0.090  0.415773  0.191794  0.392433
#   0.095  0.497971  0.210571  0.291459
#   0.100  0.580168  0.229349  0.190483
#   0.105  0.662365  0.248126  0.089509
