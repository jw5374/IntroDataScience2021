"""
Expected Inputs: 1D numpy array or list
Outputs: 2D numpy array where categories are in column 1 in ascending order, and frequency is column 2.

Created on Tue Apr  6 18:11:33 2021
"""

import numpy as np
import scipy.stats as sp


def meow(numArray):
    valsDict = {}
    for category in numArray:
        if category in valsDict.keys():
            valsDict[category] += 1
        else:
            valsDict[category] = 1
    return np.array(list(valsDict.items()))
