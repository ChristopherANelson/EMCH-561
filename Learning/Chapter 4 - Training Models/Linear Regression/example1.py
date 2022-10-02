#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:38:39 2022

@author: chris
"""




import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
X=2*np.random.rand(100,1)
y = 4+3*X+np.random.randn(100,1)

'''
plt.figure()
plt.scatter(X,y)
plt.grid(True)
plt.xlabel('$X_1$')
plt.ylabel('y')
'''
'''
Compute theta_best using the normal equation. 
    use inv() function from NumPy's linear algebra modlue (np.linalg) to compute the inverse of a matrix and the dot() for matrix multiplication'

'''

X_b = np.c_[np.ones((100,1)),X] # add x0=1 to each instance
theta_best= np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best) # ideally, theta_1 and theta_2 = 4 & 3

X_new = np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new,y_predict, 'r-')
plt.plot(X,y,'b.')
plt.axis([0, 2, 0, 15])
plt.show()


