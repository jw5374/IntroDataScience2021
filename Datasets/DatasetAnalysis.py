import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv 
ratingsMatrix = []
participantMeans = []
participantMedians = []
participantSD = []
movieMeans = []
movieMedians = []
movieSD = []
# ratings = open('movieRatingsDeidentified.csv', 'r')
# reader = csv.reader(ratings)
# header = next(reader)
# for row in reader:
#     ratingsMatrix.append(row)
# movieRatings = np.array(ratingsMatrix)

df = pd.read_csv('movieRatingsDeidentified.csv', header=0)
cols = df.select_dtypes(exclude=['float']).columns
df[cols] = df[cols].apply(pd.to_numeric, downcast = 'float', errors='coerce')



# for row in movieRatings:
#     rowCopy = row
#     fullRow = np.array(list(filter(str.strip, rowCopy))).astype(float)
#     participantMeans.append(np.mean(fullRow))
#     participantMedians.append(np.median(fullRow))
#     participantSD.append(np.std(fullRow))

# for row in movieRatings.T:
#     rowCopy = row
#     fullRow = np.array(list(filter(str.strip, rowCopy))).astype(float)
#     movieMeans.append(np.mean(fullRow))
#     movieMedians.append(np.median(fullRow))
#     movieSD.append(np.std(fullRow))

# def findMean(datalist):
#     fullList = np.array(list(filter(str.strip, datalist))).astype(float)
#     return np.mean(fullList)

# def findMedian(datalist):
#     fullList = np.array(list(filter(str.strip, datalist))).astype(float)
#     return np.median(fullList)

# def findSD(datalist):
#     fullList = np.array(list(filter(str.strip, datalist))).astype(float)
#     return np.std(fullList)

# meanOfMeans = np.nanmean(participantMeans)
# meanSD = np.nanmean(participantSD)


# print(df.corr(method='pearson'))
# print(df.corr(method='spearman'))
# print(df.T.corr(method='pearson'))
# print(df.T.corr(method='spearman'))
# print(meanOfMeans)
# print(meanSD)
# print(findSD(list(df["The Village (2004)"])))
# print(participantMeans[-1])
# print(findMean(list(df["Titanic (1997)"])))

# regr = linear_model.LinearRegression()

# dfSW = df[["Star Wars: Episode II - Attack of the Clones (2002)", "Star Wars: Episode I - The Phantom Menace (1999)"]]
# bothRatedSW = []
# for index, row in dfSW.iterrows():
#     if(row.notnull().all()):
#         bothRatedSW.append([row[0], row[1]])
# bothRatedSW = np.array(bothRatedSW)        
# dfSW = pd.DataFrame(bothRatedSW)
# dfSW.rename(columns={0: "star wars 2", 1: "star wars 1"}, inplace=True)

# regr.fit(np.array(dfSW["star wars 2"]).reshape(-1, 1), dfSW["star wars 1"])
# regrPreds = regr.predict(np.array(dfSW["star wars 2"]).reshape(-1, 1))
# print(mean_squared_error(dfSW["star wars 1"], regrPreds, squared=False))
# print(regr.coef_)
# print(r2_score(dfSW["star wars 1"], regrPreds))

# dfStartanic = df[["Titanic (1997)", "Star Wars: Episode I - The Phantom Menace (1999)"]]
# bothRatedST = []
# for index, row in dfStartanic.iterrows():
#     if(row.notnull().all()):
#         bothRatedST.append([row[0], row[1]])
# bothRatedST = np.array(bothRatedST)
# dfST = pd.DataFrame(bothRatedST)
# dfST.rename(columns={0: "titanic", 1: "star wars 1"}, inplace=True)

# regr.fit(np.array(dfST["star wars 1"]).reshape(-1, 1), dfST["titanic"])
# regrPreds = regr.predict(np.array(dfST["star wars 1"]).reshape(-1, 1))
# print(mean_squared_error(dfST["titanic"], regrPreds, squared=False))
# print(regr.coef_)
# print(r2_score(dfST["titanic"], regrPreds))

# print(listOfBothSW)

dfKillBillPulp = df[['Kill Bill: Vol. 1 (2003)', "Kill Bill: Vol. 2 (2004)", "Pulp Fiction (1994)"]]
allRatedKBP = []
for index, row in dfKillBillPulp.iterrows():
    if(row.notnull().all()):
        allRatedKBP.append([row[0], row[1], row[2]])
allRatedKBP = np.array(allRatedKBP)
dfKBPrated = pd.DataFrame(allRatedKBP)
K12paired = sp.ttest_rel(dfKBPrated[0], dfKBPrated[1])
pulpMean = dfKBPrated[2].mean()
Bill1Pulp = sp.ttest_ind(dfKBPrated[0], dfKBPrated[2])
K12indep = sp.ttest_ind(dfKBPrated[0], dfKBPrated[1])
Bill2Pulpmann = sp.mannwhitneyu(dfKBPrated[2], dfKBPrated[1])
bill1median = dfKBPrated[0].median()