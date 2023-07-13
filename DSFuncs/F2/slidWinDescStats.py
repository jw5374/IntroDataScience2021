'''
Expected Inputs: 1D numpy array for flags 1 and 2, 2D numpy array for flag 3. 
                 A flag between 1 and 3. 
                 A window size in range: 0 < windowSize < array.size.
Outputs: A 1D array with each index representing the result of the specified calculation within specified window

Standard Deviation was calculated with ddof=1 as per the example outputs. Unsure whether to use 0 or 1 as the specification sheet mentioned what the results would have been for both.

Written on March 2nd, 2021
'''

import numpy as np

def rollingOperation(numArray, windowSize, flag=1):
    splits = []    
    if(flag == 3):
        for index in range(0, (numArray.shape[0] - windowSize)+1):
            splits.append(numArray[index:index+windowSize])
    else:
        for index in range(0, (numArray.size - windowSize)+1):
            splits.append(numArray[index:index+windowSize])
    return splits

def slidWinDescStats(numArray, flag, windowSize):
    outputs = []
    splitted = rollingOperation(numArray, windowSize, flag)
    if(flag == 1):
        for split in splitted:
            outputs.append(np.mean(split))
    elif(flag == 2):
        for split in splitted:
            outputs.append(np.std(split, ddof=1))        
    elif(flag == 3):
        for split in splitted:
            outputs.append(np.corrcoef(split, rowvar=False)[0][1])
    return np.array(outputs).round(5)   
    
sample = np.array([1, 3, 5, 7, 9])
print(slidWinDescStats(sample, 1, 5))