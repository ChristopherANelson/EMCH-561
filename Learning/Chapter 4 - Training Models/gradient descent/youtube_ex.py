#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Descent Python Example
"""
import IPython as IP
IP.get_ipython().magic('reset -sf')

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn import datasets, preprocessing, linear_model, pipeline
from sklearn.linear_model import LinearRegression



plt.close('all')


def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations = 1000
    n=len(x)
    learning_rate=0.001 
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr = b_curr-learning_rate*bd
        print('m {}, b {}, cost {}, iteration {}'.format(m_curr,b_curr,cost,i))

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x, y)
