"""
Expected Inputs: 4 parameters
                priors A, B. and likelihood with range: (0.0 <= x <= 1.0)
                flag either 1 or 2 denoting simple bayesian form or explicit
                if flag is 2, prior of B should be the probability(B given not A)

Outputs: The posterior probabilty(A given B) as a floating point value with range : (0.0 <= x <= 1.0)

Created on Sat Apr 24 23:45:37 2021
"""

def bayesCalculator(priorA, priorB, likelihood, flag):
    if(flag == 1):
        return round((priorA * likelihood) / priorB, 4)
    elif(flag == 2):
        return round((priorA * likelihood) / ((likelihood * priorA) + (priorB * (1.0 - priorA))), 4)

