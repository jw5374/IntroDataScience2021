"""
Expected Inputs: 2 arguments -
                    A 1D numpy array containing real numbered values
                    A real number value in range 0.01 <= x <= 99.99
                    
Outputs: Will print and return lower and upper bound values from the array provided in the first argument
        Values will be returned in a list
    
Created on Sat May  1 14:48:19 2021
"""
import numpy as np
import math

def empiricalSampleBounds(sampleArray, massBounds):
    sampleArray.sort()
    arrayLen = sampleArray.size
    percentile = arrayLen * 0.01
    totalProbMass = 100 - massBounds
    tailMass = totalProbMass / 2
    uppertailPercentile = 100 - tailMass
    lowertailPercentile = 0 + tailMass
    upperIndex = math.floor(percentile * uppertailPercentile - 1)
    lowerIndex = math.floor(percentile * lowertailPercentile - 1)
    
    print("lowerBound = " + str(sampleArray[lowerIndex]) + ", upperBound = " + str(sampleArray[upperIndex]))
    return [sampleArray[lowerIndex], sampleArray[upperIndex]]

sampleInput1 = np.genfromtxt('./sampleInput1.csv')
empiricalSampleBounds(sampleInput1, 95)
empiricalSampleBounds(sampleInput1, 99)
empiricalSampleBounds(sampleInput1, 50)
sampleInput2 = np.genfromtxt('./sampleInput2.csv')
empiricalSampleBounds(sampleInput2, 95)
empiricalSampleBounds(sampleInput2, 99)
empiricalSampleBounds(sampleInput2, 50)