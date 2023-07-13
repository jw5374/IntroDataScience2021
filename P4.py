# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:36:46 2021

"""

import numpy as np
import pandas as pd
from scipy import stats


def splitgroups(array):
    hasGroup = []
    noGroup = []
    hasSadex = 0
    noSadex = 0
    for index in array:
        if(index[1] == 1):
            hasGroup.append(index[0])
            hasSadex += index[0]
        else:
            noGroup.append(index[0])
            noSadex += index[0]
    return [hasSadex, noSadex, hasGroup, noGroup]

    

sadex1 = np.genfromtxt('Sadex1.txt')
sadex2 = np.genfromtxt('Sadex2.txt')
sadex3 = np.genfromtxt('Sadex3.txt')
sadex4 = np.genfromtxt('Sadex4.txt')

s1 = pd.DataFrame(data=sadex1)
s2 = pd.DataFrame(data=sadex2)
s3 = pd.DataFrame(data=sadex3)
s4 = pd.DataFrame(data=sadex4)

groupSums = splitgroups(sadex1)
hasMean = groupSums[0]/(sadex1.shape[0]/2)
noHaveMean = groupSums[1]/(sadex1.shape[0]/2)
diffMean = noHaveMean - hasMean
study1 = stats.ttest_ind(groupSums[2], groupSums[3]).pvalue
diffMean2 = s2[0].mean() - s2[1].mean()
study2 = stats.ttest_ind(s2[0], s2[1]).pvalue
groupSums2 = splitgroups(sadex3)
study3 = stats.ttest_ind(groupSums2[2], groupSums2[3]).pvalue
study4 = stats.ttest_ind(s4[0], s4[1]).pvalue
standardDev4 = s4.std()
clinicalSig = (s4[0].mean() - s4[1].mean())/standardDev4[1]


