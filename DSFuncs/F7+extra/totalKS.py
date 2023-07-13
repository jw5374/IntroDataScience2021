"""
Expected Inputs: 3 arguments =>
                    Sample 1: 1D list
                    Sample 2: 1D list of same length as Sample 1
                    A flag of value either 1 or 2 (1 = classical KS, 2 = total KS normalized by joint length)
                This means that one will need to
                manually separate the columns in a csv into 2 lists before using the function

Outputs: The value of D as per the flag chosen

Created on Tue May 11 15:38:17 2021
"""
import numpy as np

def totalKS(sample1, sample2, flag):
    sorted1 = sorted(sample1)
    sorted2 = sorted(sample2)
    jointsort = []
    increment = 100.0 / len(sorted1)
    y1 = 0
    y2 = 0
    D = []
    i, j = 0, 0
    while (i < len(sorted1) and j < len(sorted2)):
        if(sorted1[i] < sorted2[j]):
            jointsort.append([1, sorted1[i]])
            i += 1
        else:
            jointsort.append([2, sorted2[j]])
            j += 1
    if(i == len(sorted1)):
        while(j < len(sorted2)):
            jointsort.append([2, sorted2[j]])
            j += 1
    else:
        while(i < len(sorted1)):
            jointsort.append([1, sorted1[i]])
            i += 1
            
    for row in jointsort:
        if(row[0] == 1):
            y1 += increment
            D.append(abs(y1 - y2))
        else:
            y2 += increment
            D.append(abs(y1 - y2))
    
    if(flag == 1):
        return round(max(D), 3)
    elif(flag == 2):
        return round((sum(D) / len(jointsort)), 3)

###### Example for formatting inputs, given example input csv's
input1 = np.genfromtxt('kSinput1.csv', delimiter=',')

sample1input1 = input1[:,0]
sample2input1 = input1[:,1]

print(totalKS(sample1input1, sample2input1, 1))
print(totalKS(sample1input1, sample2input1, 2))