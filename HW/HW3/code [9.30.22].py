# -*- coding: utf-8 -*-
"""
HW3 - 10/4/2022
Team 3 - Christopher Nelson | [NAME] | [NAME] | [NAME]

"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn import datasets, preprocessing, linear_model, pipeline
from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import PolynomialFeatures

plt.close('all')


#%% LOAD DATA

X=np.loadtxt('supporting_files/X_DATA.txt').reshape(-1,1)

y=np.loadtxt('supporting_files/Y_DATA.txt').reshape(-1,1)

X_train_data = np.loadtxt('supporting_files/X_train.txt').reshape(-1,1)

X_validation_data = np.loadtxt('supporting_files/X_validation.txt').reshape(-1,1)


y_train_data = np.loadtxt('supporting_files/y_train.txt').reshape(-1,1)

y_validation_data = np.loadtxt('supporting_files/y_validation.txt').reshape(-1,1)








#%% GENERATE PLOT 1


plt.figure()
#plt.plot(X,y,'b.')
plt.scatter(X_train_data,y_train_data, label='Training Set')
plt.scatter(X_validation_data,y_validation_data, label='Validation Set', marker='X')

plt.legend()
plt.show()






#%% POLY REGRESSION 20
poly_features = PolynomialFeatures(degree=2,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)


Coef=reg.coef_.T
intercept=reg.intercept_
#pred_label=str(Coef[1]) +'X^2 + ' + str(Coef[0]) + 'X + ' + str(intercept)




#%% GENERATE PLOT 2

plt.figure()
plt.plot(X_vals, y_vals, color='r', label='Polynomial [2]')
plt.scatter(X_train_data,y_train_data, label='Training Set')
plt.scatter(X_validation_data,y_validation_data, label='Validation Set', marker='X')
plt.ylim(0,10)



#%%

poly_features = PolynomialFeatures(degree=20,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)


Coef=reg.coef_.T
intercept=reg.intercept_
#pred_label=str(Coef[1]) +'X^2 + ' + str(Coef[0]) + 'X + ' + str(intercept)


#%%

plt.plot(X_vals, y_vals, color='g', label='Polynomial [20]', linestyle='--')




plt.legend()