"""
Created on Sat May  8 12:00:42 2021
"""
import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import meow

df = pd.read_csv('middleSchoolData.csv', header=0)
# cols = df.select_dtypes(exclude=['float']).columns
# df[cols] = df[cols].apply(pd.to_numeric, downcast = 'float', errors='coerce')

#%% Question 1
plot1 = plt.figure(1)
plt.scatter(df['applications'], df['acceptances'])
plt.xlabel('application count')
plt.ylabel('acceptances')
acceptancecorr = df['applications'].corr(df['acceptances'], method='spearman')
acceptancecorrpears = df['applications'].corr(df['acceptances'], method='pearson')
acceptCOD = acceptancecorrpears ** 2
plt.savefig('acceptanceApplicationsplot.png')

####

#%% Question 2
regr = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
applications = df['applications']
def rateConv(array, divArray):
    divide = divArray
    rates = []
    for index in range(divide.size):
        if((array[index] == 0) and divide[index] == 0):
            rates.append(float('nan'))
        else:
            rates.append((array[index] / divide[index]))
    return rates
applicationRate = pd.Series(rateConv(applications, df['school_size']))
df['application_rate'] = pd.DataFrame(applicationRate)
dfRate = df[['acceptances', 'application_rate']]
dfAccept = df[['acceptances', 'applications']]
ratenotnans = []
for index, row in dfRate.iterrows():
    if(row.notnull().all()):
        ratenotnans.append([row[0], row[1]])
dfRate = pd.DataFrame(ratenotnans)
regr.fit(np.array(dfRate[1]).reshape(-1,1), dfRate[0])
regrPreds = regr.predict(np.array(dfRate[1]).reshape(-1,1))
print(mean_squared_error(dfRate[0], regrPreds, squared=False))
print(regr.coef_)
print(r2_score(dfRate[0], regrPreds))

regr2.fit(np.array(dfAccept['applications']).reshape(-1,1), dfAccept['acceptances'])
regr2Preds = regr2.predict(np.array(dfAccept['applications']).reshape(-1,1))
print(mean_squared_error(dfAccept['acceptances'], regr2Preds, squared=False))
print(regr2.coef_)
print(r2_score(dfAccept['acceptances'], regr2Preds))

appRatecorr = df['acceptances'].corr(applicationRate, method='spearman')
appRatepears = df['acceptances'].corr(applicationRate, method='pearson')

appRateCOD = appRatepears ** 2
plot2 = plt.figure(2)
plt.scatter(applicationRate, df['acceptances'])
plt.xlabel('application rate')
plt.ylabel('acceptances')
plt.savefig('applicationRateplot.png')

####

#%% Question 3
acceptRate = pd.Series(rateConv(df['acceptances'], df['applications']))
studentOdds = []
for index in range(acceptRate.size):
    prob = applicationRate[index] * acceptRate[index]
    if(not (math.isnan(prob))):
        studentOdds.append(prob / (1-prob))
    else:
        studentOdds.append(0)
df['student_odds'] = pd.DataFrame(studentOdds)
plot3 = plt.figure(3)
plt.bar(df.index.values.tolist(), height=pd.Series(studentOdds), width=5, tick_label=None)
plt.xlabel('schools')
plt.ylabel('odds')
plt.savefig('acceptOddsplot.png')
maxOdds = df['student_odds'].idxmax()
####

#%% Question 4
dfClimate = df[[
    'rigorous_instruction', 
    'collaborative_teachers', 
    'supportive_environment', 
    'effective_school_leadership', 
    'strong_family_community_ties', 
    'trust'
    ]]

dfPerformance = df[[
    'student_achievement',
    'reading_scores_exceed',
    'math_scores_exceed'
    ]]

def remNAN(mat):
    nans=[]
    for index, row in mat.iterrows():
        if(not (row.notnull().all())):
            nans.append(index)
    mat.drop(nans, inplace=True)
    
dfBothCliPerf = pd.concat([dfClimate, dfPerformance], axis=1)
remNAN(dfBothCliPerf)
dfNoNANClim = dfBothCliPerf[[
    'rigorous_instruction', 
    'collaborative_teachers', 
    'supportive_environment', 
    'effective_school_leadership', 
    'strong_family_community_ties', 
    'trust'
    ]]
dfNoNANPerf = dfBothCliPerf[[
    'student_achievement',
    'reading_scores_exceed',
    'math_scores_exceed'
    ]]
dfOnlyRatings = dfBothCliPerf[[
    'rigorous_instruction', 
    'collaborative_teachers', 
    'supportive_environment', 
    'effective_school_leadership', 
    'strong_family_community_ties', 
    'trust',
    'student_achievement'
    ]]

correlationMat = dfBothCliPerf.corr(method='pearson')
zscored = sp.stats.zscore(dfBothCliPerf)
pca = PCA()
pca.fit(zscored)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscored)
covarExplained = eigVals/sum(eigVals)*100

zscored2 = sp.stats.zscore(dfNoNANPerf)
pca2 = PCA()
pca2.fit(zscored2)
eigVals2 = pca2.explained_variance_
loadings2 = pca2.components_
rotatedData2 = pca2.fit_transform(zscored2)
covarExplained2 = eigVals2/sum(eigVals2)*100

zscored3 = sp.stats.zscore(dfNoNANClim)
pca3 = PCA()
pca3.fit(zscored3)
eigVals3 = pca3.explained_variance_
loadings3 = pca3.components_
rotatedData3 = pca3.fit_transform(zscored3)
covarExplained3 = eigVals3/sum(eigVals3)*100

correlationComps = np.corrcoef(rotatedData2[:,0], rotatedData3[:,0])

zscored4 = sp.stats.zscore(dfOnlyRatings)
pca4 = PCA()
pca4.fit(zscored4)
eigVals4 = pca4.explained_variance_
loadings4 = pca4.components_
rotatedData4 = pca4.fit_transform(zscored4)
covarExplained4 = eigVals4/sum(eigVals4)*100
correlationMatRatings = dfOnlyRatings.corr(method='pearson')

plot4 = plt.figure(4)
plt.imshow(correlationMat, aspect='auto')
plt.colorbar()

# plot5 = plt.figure(5)
# numClasses = 9
# plt.bar(np.linspace(0,8, 9), height=eigVals)
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.plot([0,numClasses],[1,1],color='red',linewidth=1)

plot6 = plt.figure(6)
numClasses2 = 6
plt.bar(np.linspace(0,5,6), height=eigVals3)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses2],[1,1],color='red',linewidth=1)

plot7 = plt.figure(7)
numClasses3 = 3
plt.bar(np.linspace(0,2,3), height=eigVals2)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses3],[1,1],color='red',linewidth=1)

plot8 = plt.figure(12)
numClasses4 = 7
plt.bar(np.linspace(0,6,7), height=eigVals4)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses4],[1,1],color='red',linewidth=1)

# plot9 = plt.figure(8)
# plt.bar(np.linspace(0,8,9),loadings[:,2])
# plt.xlabel('Question')
# plt.ylabel('Loading')

plot10 = plt.figure(9)
plt.bar(np.linspace(0,5,6),loadings3[:,0])
plt.xlabel('Question')
plt.ylabel('Loading')

plot11 = plt.figure(10)
plt.bar(np.linspace(0,2,3),loadings2[:,0])
plt.xlabel('Question')
plt.ylabel('Loading')

plot12 = plt.figure(13)
plt.bar(np.linspace(0,6,7),loadings4[:,1])
plt.xlabel('Question')
plt.ylabel('Loading')

plot13= plt.figure(11)
plt.plot(rotatedData4[:,0],rotatedData4[:,1],'o',markersize=1)

plot14 = plt.figure(14)
plt.imshow(correlationComps, aspect='auto')
plt.colorbar()
####

#%% Question 5
dfSpendingSize = df[[
        'per_pupil_spending',
        'avg_class_size',
        'poverty_percent',
        'student_achievement'
    ]]
remNAN(dfSpendingSize)
# model = sm.OLS(dfSpendingSize['student_achievement'], dfSpendingSize['avg_class_size']).fit()
# modelPredict = model.predict(dfSpendingSize['avg_class_size'])
# print(model.summary())

spendingMed = dfSpendingSize['per_pupil_spending'].median()
classMed = dfSpendingSize['avg_class_size'].median()
povertyMed = dfSpendingSize['poverty_percent'].median()
def categorizeData(dataseries, threshold):
    categorizedlist = []
    for value in dataseries:
        if value >= threshold:
            categorizedlist.append(2)
        else:
            categorizedlist.append(1)
    return categorizedlist
spendingCats = categorizeData(dfSpendingSize['per_pupil_spending'], spendingMed)
sizeCats = categorizeData(dfSpendingSize['avg_class_size'], classMed)
povCats = categorizeData(dfSpendingSize['poverty_percent'], 50.0)
richAchieve = []
poorAchieve = []
achieves = np.array(dfSpendingSize['student_achievement'])
for index in range(len(spendingCats)):
    if spendingCats[index] == 1:
        poorAchieve.append(achieves[index])
    else:
        richAchieve.append(achieves[index])
histo = pd.DataFrame(meow.meow(dfSpendingSize["student_achievement"]))
histo.sort_values(0, ascending=True, inplace=True)
spendgraph = plt.figure(15)
plt.bar(np.linspace(1, 225, 225), histo[1])
#plt.bar(["rich", 'poor'], height=[np.mean(richAchieve), np.mean(poorAchieve)])

####

#%% Question 6
# model2 = ols('student_achievement ~ per_pupil_spending + avg_class_size + per_pupil_spending:avg_class_size', data=dfSpendingSize).fit() 
# anova_table2 = sm.stats.anova_lm(model2, typ=2) #Create the ANOVA table. Residual = Within
# print(anova_table2) #Show the ANOVA table

# fig = meansPlot(x=dfSpendingSize['per_pupil_spending'], trace=dfSpendingSize['avg_class_size'], response=dfSpendingSize['student_achievement'])

bigAchieve = []
smallAchieve = []
for index in range(len(sizeCats)):
    if sizeCats[index] == 1:
        smallAchieve.append(achieves[index])
    else:
        bigAchieve.append(achieves[index])

sizegraph = plt.figure(16)
plt.bar(["big", 'small'], height=[np.mean(bigAchieve), np.mean(smallAchieve)])



####

#%% Question 7
acceptances = df[['acceptances', 'school_name']]
sortedAccept = acceptances.sort_values(by=['acceptances'], ascending=False)
totalAccepts = acceptances['acceptances'].sum()
arraySorted = np.array(sortedAccept)
schoolCount = 0
acceptanceThreshold = 0
for row in arraySorted:
    if(acceptanceThreshold < 4015):
        acceptanceThreshold += row[0]
        schoolCount += 1
    else:
        break


AcceptanceBarGraph = plt.figure(50)
plt.bar(acceptances.index.values.tolist(), height=sortedAccept['acceptances'], width=5)
plt.xlabel('schools')
plt.ylabel('acceptances')

####

#%% Question 8
dfFactors = df[[
        'applications',
        'acceptances',
        'per_pupil_spending',
        'avg_class_size',
        'asian_percent',
        'black_percent',
        'hispanic_percent',
        'multiple_percent',
        'white_percent',
        'rigorous_instruction', 
        'collaborative_teachers', 
        'supportive_environment', 
        'effective_school_leadership', 
        'strong_family_community_ties', 
        'trust',
        'disability_percent',
        'poverty_percent',
        'ESL_percent',
        'school_size',
        'student_achievement',
        'reading_scores_exceed',
        'math_scores_exceed',
        'application_rate',
        'student_odds'
    ]]
remNAN(dfFactors)
factorMat = dfFactors.corr(method='pearson')
def appendEqualCountsClass(df, class_name, feature, num_bins, labels):
    '''Append a new class feature named 'class_name' based on a split of 'feature' into clases with equal sample points.  Class names are in 'labels'.'''

    # Compute the bin boundaries
    percentiles = np.linspace(0,100,num_bins+1)
    bins = np.percentile(df[feature],percentiles)

    # Split the data into bins
    n = pd.cut(df[feature], bins = bins, labels=labels, include_lowest=True)

    # Add the new binned feature to a copy of the data
    c = df.copy()
    c[class_name] = n
    return c

dfFactors = appendEqualCountsClass(dfFactors, "accepted", "student_odds", 2, ["L","H"])
dfFactors2 = appendEqualCountsClass(dfFactors, "achievementlevel", "student_achievement", 2, ["L","H"])
y = dfFactors['accepted']
X = dfFactors[[
        'applications',
        'per_pupil_spending',
        'avg_class_size',
        'asian_percent',
        'black_percent',
        'hispanic_percent',
        'multiple_percent',
        'white_percent',
        'rigorous_instruction', 
        'collaborative_teachers', 
        'supportive_environment', 
        'effective_school_leadership', 
        'strong_family_community_ties', 
        'trust',
        'disability_percent',
        'poverty_percent',
        'ESL_percent',
        'school_size',
        'student_achievement',
        'reading_scores_exceed',
        'math_scores_exceed',
        'application_rate'
    ]]

y2 = dfFactors2['achievementlevel']
X2 = dfFactors2[[
        'per_pupil_spending',
        'avg_class_size',
        'asian_percent',
        'black_percent',
        'hispanic_percent',
        'multiple_percent',
        'white_percent',
        'rigorous_instruction', 
        'collaborative_teachers', 
        'supportive_environment', 
        'effective_school_leadership', 
        'strong_family_community_ties', 
        'trust',
        'disability_percent',
        'poverty_percent',
        'ESL_percent',
        'school_size',
        'reading_scores_exceed',
        'math_scores_exceed'
    ]]

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
rescaledX2 = scaler.fit_transform(X2)

X = pd.DataFrame(rescaledX, columns=X.columns)
X2 = pd.DataFrame(rescaledX2, columns=X2.columns)

test_size = 0.5
seed = 12345
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size=test_size, random_state=seed)
model_lr = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
model_lr.fit(X_train, Y_train)
predictions_lr = model_lr.predict(X_train)
print("LogisticRegression", accuracy_score(Y_train, predictions_lr))
predictions_lr = model_lr.predict(X_test)
print("LogisticRegression", accuracy_score(Y_test, predictions_lr))

def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''
    
    # Get a list of class names
    numclasses = len(model.classes_)
    if len(model.classes_)==2:
        classes =  [model.classes_[1]] # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_
    
    # Create a plot for each class
    for i,c in enumerate(classes):
        # Plot the coefficients as bars
        fig = plt.figure(figsize=(8,len(column_names)/3))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16)
        rects = plt.barh(column_names, model.coef_[i],color="lightblue")
        
        # Annotate the bars with the coefficient values
        for rect in rects:
            width = round(rect.get_width(),4)
            plt.gca().annotate('  {}  '.format(width),
                        xy=(0, rect.get_y()),
                        xytext=(0,2),  
                        textcoords="offset points",  
                        ha='left' if width<0 else 'right', va='bottom')        
        plt.show()
        #for pair in zip(X.columns, model_lr.coef_[i]):
        #    print (pair)
logisticRegressionSummary(model_lr, X.columns)

model_lr2 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
model_lr2.fit(X_train2, Y_train2)
predictions_lr2 = model_lr2.predict(X_train2)
print("LogisticRegression2", accuracy_score(Y_train2, predictions_lr2))
predictions_lr2 = model_lr2.predict(X_test2)
print("LogisticRegression2", accuracy_score(Y_test2, predictions_lr2))
logisticRegressionSummary(model_lr2, X2.columns)



####
