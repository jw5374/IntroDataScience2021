# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:44:17 2021

@author: aweso
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

klonopindata = np.genfromtxt("klonopin.txt") 
dose1 = []
dose2 = []
dose3 = []
dose4 = []
dose5 = []
dose6 = []
for row in klonopindata:
    if row[1] == 0:
        dose1.append(row[0])
    elif row[1] == 0.1:
        dose2.append(row[0])
    elif row[1] == 0.2:
        dose3.append(row[0])
    elif row[1] == 0.5:
        dose4.append(row[0])
    elif row[1] == 1:
        dose5.append(row[0])
    elif row[1] == 2:
        dose6.append(row[0])
klonoanova = stats.f_oneway(dose1, dose2, dose3, dose4, dose5, dose6)
klonomeans = [np.mean(dose1), np.mean(dose2), np.mean(dose3), np.mean(dose4), np.mean(dose5), np.mean(dose6)]


filename = 'socialstress.txt'
df = pd.read_fwf(filename, header=None) #Import the data from the csv file into a dataframe
df.info() #What is the structure of the data frame?
df.rename(columns={0: 'Value', 1: 'X1', 2: 'X2', 3: "X3", 4: "X4"}, inplace=True)

model = ols('Value ~ X1 + X2 + X3 + X4 + X1:X2 + X1:X3 + X1:X4 + X2:X3 + X2:X4 + X3:X4 + X1:X2:X3 + X1:X3:X4 + X1:X2:X4 + X2:X3:X4 + X1:X2:X3:X4', data=df).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
anova_table = sm.stats.anova_lm(model, typ=3) #Create the ANOVA table. Residual = Within
print(anova_table) #Show the ANOVA table

happiness = np.genfromtxt('happiness.txt')
def splitgroups(array):
    hasGroup = []
    noGroup = []
    for index in array:
        if(index[1] == 1):
            hasGroup.append(index[0])
        else:
            noGroup.append(index[0])
    return [hasGroup, noGroup]

lotterygroups = splitgroups(happiness)
mannwhitneylotter = stats.mannwhitneyu(lotterygroups[0], lotterygroups[1])