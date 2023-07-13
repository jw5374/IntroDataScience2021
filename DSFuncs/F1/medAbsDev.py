"""
Input: 1D numpy array
Output: Median Absolute Deviation of the dataset

Written on: February 24th, 2021
"""
import numpy as np

def findMed(numArray):
    if(numArray.size % 2 == 0):
        median = (numArray[(numArray.size // 2) - 1] + numArray[numArray.size // 2]) / 2
    else:
        median = numArray[(numArray.size // 2)]
    return median

def medAbsDev_nonMedianFunction(numArray):
    writtenBy = "Joseph Wang"
    sortedArray = np.sort(numArray, axis=0)
    median = findMed(sortedArray)
    devArray = []
    for num in numArray:
        devArray.append(abs(num - median))
    sortedDev = np.sort(np.array(devArray), axis=0)
    return findMed(sortedDev)

def medAbsDev(numArray):
    writtenBy = "Joseph Wang"
    median = np.median(numArray, axis=0)
    devArray = []
    for num in numArray:
        devArray.append(abs(num - median))
    return np.median(np.array(devArray), axis=0)

data = np.array([4.5, 3.2, 1.5, 5.7, 9.3, 2.2, 6.9])
data_2 = np.array([1, 3, 5, 7, 9])
print(medAbsDev(data_2))
print(medAbsDev(data))

    