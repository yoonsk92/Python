import numpy as np
import seaborn as sns
import pandas as pd

correlation = np.array([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
                        [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
                        [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
                        [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
                        [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
                        [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
                        [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])

vol = np.array([0.16, 0.203, 0.248, 0.271, 0.21, 0.20, 0.187])

#seq: Aus, Cad, France, German, Jap, UK, USA
N,M = correlation.shape

def cov(cor,vol):
    cov = np.zeros((7,7))
    for i in range(M):
        for j in range(N):
            cov[i,j] = cor[i,j]*(vol[i]*vol[j])
    return cov

cov = cov(correlation, vol)

#The initial point in A_1
mu_0 = np.array([0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
w_0 = np.linalg.inv(2.5*cov).dot(mu_0)
mu_1 = np.array([0.07, 0.07, 0.045, 0.095, 0.07, 0.045, 0.07])
w_1 = np.linalg.inv(2.5*cov).dot(mu_1)
df1 = pd.DataFrame(w_0, columns = ['weight'], index = ['AUS', 'CAD', 'FR', 'GER', 'JAP', 'UK', 'US'])
df2 = pd.DataFrame(w_1, columns = ['weight'], index = ['AUS', 'CAD', 'FR', 'GER', 'JAP', 'UK', 'US'])

frames = [df1,df2]
result = pd.concat(frames, keys = ['x','y']).reset_index()
ax = sns.barplot(x='level_1', y= 'weight', hue = 'level_0', data = result)










