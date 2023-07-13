# This function outputs the number of columns for input 'input_array'
import numpy as np
def width_func(input_array): # and use it as the input to our function
    step1 = np.size(input_array) # total number of elements
    step2 = len(input_array) # total number of rows
    return int(step1/step2) # return total number of elements divided by number of rows