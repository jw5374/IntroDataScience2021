"""
Expected Inputs: A 1D numpy array of data

Outputs: A list of length 2 containing the imputed dataset, and a list of the imputed values
        This is a nondestructive function, the original dataset is left untouched

Created on Tue May 11 13:42:00 2021
"""
from itertools import groupby
from operator import itemgetter
import numpy as np

def checkconsec(nanlist):
    rangeslist = []
    ## taken from itertools documentation: https://docs.python.org/2.6/library/itertools.html#examples
    for k, g in groupby(enumerate(nanlist[0]), lambda ix : ix[0] - ix[1]):
        rangeslist.append(list(map(itemgetter(1), g)))
    return rangeslist

def imputationByInterpolation(data):
    dataCopy = data.copy()
    nans = np.where(np.isnan(dataCopy))
    ranges = checkconsec(nans)
    imputes = []
    for row in ranges:
        imputeind = 0
        start = row[0]-1
        end = row[-1]+1
        impute = list(np.linspace(dataCopy[start], dataCopy[end], (end-start)+1))
        del impute[0]
        del impute[-1]
        imputes.append(impute)
        for value in row:
            dataCopy[value] = impute[imputeind].round(3)
            imputeind += 1       
    return [dataCopy, imputes]


sampleinput1 = np.genfromtxt('sampleImput1.csv')
sampleinput2 = np.genfromtxt('sampleImput2.csv')
sampleinput3 = np.genfromtxt('sampleImput3.csv')
output1 = imputationByInterpolation(sampleinput1)
output2 = imputationByInterpolation(sampleinput2)
output3 = imputationByInterpolation(sampleinput3)
