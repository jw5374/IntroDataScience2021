"""
Created on Mon Mar 15 21:44:33 2021

"""
import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp
from sklearn.linear_model import LinearRegression 

alienData = np.genfromtxt("kepler.txt")
df = pd.DataFrame(data=alienData)

df.rename(columns={0: 'caste', 1: 'iq', 2: 'brainMass', 3: 'hoursWorked', 4: 'income'}, inplace=True)

lin_reg = sp.stats.linregress(df['brainMass'], df['iq'])
part_corr = pg.partial_corr(data=df, x='caste', y='income', covar=['iq', 'hoursWorked'], method='pearson').r


print(df['iq'].corr(df['caste'], method='pearson'))
print(pg.partial_corr(data=df, x='caste', y='iq', covar='brainMass', method='pearson').r)
print((df['iq'].corr(df['brainMass'], method='pearson'))**2)
print(df['income'].corr(df['caste'], method='pearson'))
print(df['income'].corr(df['hoursWorked'], method='pearson')**2)
print(part_corr)
multi_reg = LinearRegression().fit(df[['iq', 'hoursWorked']], df['income'])
print(multi_reg.score(df[['iq', 'hoursWorked']], df['income'])**2)
print(multi_reg.predict([[120, 50]]))

def predictLinReg(value):
    return (lin_reg.slope*value) + lin_reg.intercept

