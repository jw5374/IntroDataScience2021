# This function outputs the digitsum for input 'num'
def digitsum_func(num):
    temp = str(num) # convert it to a string
    while len(temp) > 1: # as long as it is not a single digit
        temp2 = 0 # initialize sum
        for i in range(len(temp)): # go through all digits
            temp2 = temp2 + int(temp[i]) # add the ith digit as a number to the cumulative sum
        return temp2 