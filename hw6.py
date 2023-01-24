# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:28:12 2022

@author: Shade
"""

from random import uniform
import math
import numpy as np
import sys
def computePi (i,precision,pi_calc,in_circle,trials,avg_pi,successful_attempts,average_pi):
    while i < 100:
        while (abs(np.pi-pi_calc) > precision):
            # generate random numbers between 0-1 a total of 10000 iterations
            if trials == 10000:
                in_circle = 0
                trials = 0
                pi_calc = 0.0
                i += 1
                # If this section is reached calls function again with reset values and i incremented for the first while loop check
                return(computePi (i,precision,pi_calc,in_circle,trials,avg_pi,successful_attempts,average_pi))
            x_value = uniform(0.0,1.0)
            y_value = uniform(0.0,1.0)
    		# check if R is inside circle or outside circle         
            if np.hypot (x_value,y_value) <= 1:
                in_circle += 1
            trials += 1
            pi_calc = ((4.0 * in_circle) / trials)  
        # counts each successful attempt and appends the calculated value into the the previous ones then takes the average 
        successful_attempts += 1
        avg_pi = np.append(avg_pi,pi_calc)
        average_pi = (sum(avg_pi))/successful_attempts
        in_circle = 0
        trials = 0
        pi_calc = 0.0
        i += 1
        # return the average value of pi along with the successful attempts per precision
    return(average_pi, successful_attempts)


for precision in [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]:
    # initialize all values to be sent into the calculate Pi function
    i = 0
    pi_calc = 0
    in_circle = 0
    trials = 0
    avg_pi = []
    successful_attempts = 0
    average_pi = 0.0
    # call function
    Pi_result = computePi(i,precision,pi_calc,in_circle,trials,avg_pi,successful_attempts,average_pi)
    # prints out the recision how many successful attempts there were and the value calculated
    if not Pi_result[1] == 0:
        print("%s success %d times %9.20f\n" %(precision, Pi_result[1], Pi_result[0]))
    else:
        print("%s no success\n" %precision)