import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("Sample data.csv").set_index('Year')

def rreturn(colname):
    rates = pd.DataFrame()
    index = df.columns.get_loc(colname)
    for row in range(len(df)-1):
        rates.loc[row,colname] = ((df.iloc[row+1,index] - df.iloc[row,index]) / (df.iloc[row,index]))*100
    return rates   
rates = pd.concat([rreturn('Stocks'),rreturn('Bonds'),rreturn('MM')], axis = 1)

def mean_method(x, colname):
    if x == 'arith':
        a = (100 + rreturn(colname)) / 100
        arith_mean = (np.mean(a) - 1) * 100
        return arith_mean
    elif x == 'geo':
        g = (100 + rreturn(colname)) / 100
        geo_mean = pd.Series((stats.gmean(g)-1) * 100)
        return geo_mean
arthdf = pd.concat([mean_method('arith','Stocks'), mean_method('arith','Bonds'), mean_method('arith','MM')])
geodf = pd.concat([mean_method('geo','Stocks'), mean_method('geo','Bonds'), mean_method('geo','MM')]) / 100
geodf.index = ['Stocks','Bonds','MM']

def cov(x1,x2):
    a = np.sum((rates[x1]-arthdf[x1])*(rates[x2]-arthdf[x2]))*(1/len(rates))*1/10000
    return a

def covM(df):
    temp = df.columns.values
    b = pd.DataFrame(columns = temp, index = range(len(df.columns)))
    for col in range(len(df.columns)):
        for row in range(len(df.columns)):
            b.iloc[row,col] = cov(df.columns[row], df.columns[col])
    return b

from pyomo.environ import *
def MVO(R):
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
    model.con1 = Constraint(expr = sum(geodf.loc[j]*model.x[j] for j in X) >= R)
    model.con2 = Constraint(expr = sum(model.x[j] for j in X) == 1)
    
    op.solve(model)
    results.loc[R] = [model.x['Stocks'].value,
                        model.x['Bonds'].value,
                        model.x['MM'].value]
    return results

def rdf(Rs,Re, incr):
    ER = np.arange((Rs/100), ((Re+0.1)/100), incr/100)
    rates_df = pd.DataFrame(columns = df.columns)
    for ers in ER:
        rates_df.loc[ers] = MVO(ers).loc[ers]
    return rates_df

rdf(6.5, 10.5, 0.5)


