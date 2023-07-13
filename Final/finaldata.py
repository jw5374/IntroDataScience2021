"""
Created on Wed May 12 23:00:42 2021

"""
import numpy as np
import pandas as pd
import scipy.stats as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from DataScience.DSFuncs.F4 import meow
import matplotlib.pyplot as plt



thanksgiving = pd.read_csv('How much you enjoy Thanksgiving based on turkey, family, and cooking.csv', header=0)
turkeylist = []
famfeellist = []
rolelist = []
thanksgiving.rename(columns={'Love Thanksgiving': 'LoveThanksgiving','Feeling about turkey': 'Feelingaboutturkey','Feeling about family': 'Feelingaboutfamily','Role ' : 'Role'}, inplace=True)
for value in thanksgiving['Feelingaboutturkey']:
    if(value == "Don't like turkey"):
        turkeylist.append(1)
    else: 
        turkeylist.append(2)
        
for value in thanksgiving['Feelingaboutfamily']:
    if(value == "They're annoying"):
        famfeellist.append(1)
    else: 
        famfeellist.append(2)
        
for value in thanksgiving['Role']:
    if(value == "Have to cook"):
        rolelist.append(1)
    else: 
        rolelist.append(2)
thanksgiving['Feelingaboutturkey'] = turkeylist
thanksgiving['Feelingaboutfamily'] = famfeellist
thanksgiving['Role'] = rolelist

model = ols('LoveThanksgiving ~ Feelingaboutturkey + Feelingaboutfamily + Role + Feelingaboutturkey:Feelingaboutfamily + Feelingaboutturkey:Role + Feelingaboutfamily:Role + Feelingaboutturkey:Feelingaboutfamily:Role', data=thanksgiving).fit() 
anova_table = sm.stats.anova_lm(model, typ=3) #Create the ANOVA table. Residual = Within
print(anova_table) #Show the ANOVA table
print(sp.stats.kruskal(thanksgiving['LoveThanksgiving'], thanksgiving['Feelingaboutturkey'], thanksgiving['Feelingaboutfamily'], thanksgiving['Role']))
print(sp.stats.f_oneway(thanksgiving['LoveThanksgiving'], thanksgiving['Feelingaboutturkey'], thanksgiving['Feelingaboutfamily'], thanksgiving['Role']))

print(sp.chisquare([76,44], [60, 60]))

college = pd.read_csv('College Success.csv', header=0)
collegemodel = sm.OLS(college['gpa'], college[['hsm', 'hss', 'hse', 'satm', 'satv']]).fit()
collegepredict = collegemodel.predict(college[['hsm', 'hss', 'hse', 'satm', 'satv']])

print(collegemodel.summary())

collegecorr = college.corr(method='pearson')

sadex2 = pd.read_csv('Assignment 7 - Sadex2.csv', header=0)
sadextest = sp.ttest_rel(sadex2['Before'], sadex2['After'])

beer = pd.read_csv('Beer Goggles.csv', header=0)
model2 = ols('Attractiveness ~ FaceType + Alcohol + FaceType:Alcohol', data=beer).fit() 
anova_table2 = sm.stats.anova_lm(model2, typ=2) #Create the ANOVA table. Residual = Within
print(anova_table2) #Show the ANOVA table

dark = pd.read_csv('DarkTriad _1_.csv', header=0)
print(dark[['Narcissism', 'Machiavellianism']].corr(method='pearson'))
machCount = 0
for value in dark['Machiavellianism']:
    if (value > 80):
        machCount += 1
print(machCount / dark['Machiavellianism'].size)

answers = pd.read_csv('jw5374.csv', skiprows=4)
cats = pd.DataFrame(meow.meow(np.array(answers['Response'])))
cats.sort_values(0, ascending=True, inplace=True)
cats[1] = pd.to_numeric(cats[1])

plt.bar(cats[0], height=cats[1], bottom=0)
