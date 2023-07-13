"""
Expected Inputs: A 2D numpy array with 2 columns, column 1 are predictions and column 2 are measurements
                and a flag denoting the power/root used for calculation ( >= 1 )
Outputs: A single scalar for the error being calculated based on flag


Created on Thu Mar 25 15:42:50 2021

"""

import numpy as np

def normalizedError(numMatrix, flag):
    rows = numMatrix.shape[0]   
    summation = 0
    for row in numMatrix:
        summation += (abs((row[0] - row[1]))) ** flag
    root = 1/flag
    return ((summation/rows) ** root).round(4)

exampleInput = np.array([[1, 2], [2, 4], [3, 1], [4, 6], [5, 2]])
exampleInput2 = np.array([[0, 50], [10, 5], [5, 10]])
print(normalizedError(exampleInput2, 3))