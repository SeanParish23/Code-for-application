# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 00:00:24 2022

@author: Shade
"""
"""
Make sure to run each cell individually 
"""

#%% Function definitions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

range_t = np.arange(0,7,.01)

def problem1(y,t):
    
    dydt_1 = np.cos(t)
    
    return (dydt_1)

def problem2(y,t):
    
    dydt_2 = -y + (t**2.0) * (np.exp(-2.0*t)) + 10
    
    return (dydt_2)

def problem3(y,t):
    
    
    return(y[1],-4*y[1] - 4*y[0] + 25*np.cos(t) + 25*np.sin(t))
#%% problem 1
y_0 = 1

y = odeint(problem1, y_0, range_t)

plt.plot(range_t, y)
plt.xlabel('time')
plt.ylabel('y_1(t)')
plt.show()

#%% problem 2

y_0 = 0

y = odeint(problem2, y_0, range_t)

plt.plot(range_t, y)
plt.xlabel('time')
plt.ylabel('y_2(t)')
plt.show()

#%% problem 3
y_0 = [1,1]

y = odeint(problem3, y_0, range_t)

y_deriv = y[:,1]
y_one = y[:,0]


plt.plot(range_t, y_one)
plt.plot(range_t, y_deriv)
plt.xlabel('time')
plt.ylabel("y_3(t) & y'_3(t)")
plt.show()