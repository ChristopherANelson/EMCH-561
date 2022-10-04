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
from sklearn.linear_model import LinearRegression, Ridge


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

plt.close('all')

#%% LOAD DATA

X=np.loadtxt('supporting_files/X_DATA.txt').reshape(-1,1)

y=np.loadtxt('supporting_files/Y_DATA.txt').reshape(-1,1)

X_train = np.loadtxt('supporting_files/X_train.txt').reshape(-1,1)

X_val = np.loadtxt('supporting_files/X_validation.txt').reshape(-1,1)

y_train = np.loadtxt('supporting_files/y_train.txt').reshape(-1,1)

y_val = np.loadtxt('supporting_files/y_validation.txt').reshape(-1,1)

plt.figure("fig1")
plt.scatter(X_train,y_train, label='Training Set')
plt.scatter(X_val,y_val, label='Validation Set', marker='X')
plt.ylim(0,10)

#%% POLY REGRESSION 2
poly_features = PolynomialFeatures(degree=2,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='red', label='Polynomial [2]')


#%% POLY REGRESSION 16
poly_features = PolynomialFeatures(degree=16,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='purple', label='Polynomial [16]', linestyle='dashdot')


#%% Linear Regression MSE
model = LinearRegression()

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

model = Pipeline((
    ('poly_features', PolynomialFeatures(degree=16, include_bias=False)),
    ('lin_reg', LinearRegression(),
    )))

train_errors, val_errors = [], []
for i in range(1,len(X_train)):
    model.fit(X_train[:i],y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    
    #compute the error for the trained model
    mse_train = mean_absolute_error(y_train[:i],y_train_predict)
    train_errors.append(mse_train)
    
    #compute the error for the validation model
    mse_val = mean_absolute_error(y_val, y_val_predict)
    val_errors.append(mse_val)
    
    plt.figure('test model')
    plt.scatter(X, y, s=2, label='data')
    plt.scatter(X_train[:1], y_train[:1], s=2, label='data in training set')
    plt.scatter(X_val[:1], y_val[:1], marker='s', label='validation set')
    y_model=model.predict(X_model)
    plt.plot(X_model,y_model,'r--',label='model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('test_plots/linear_model'+str(i))
    plt.close('test model')
    
plt.figure("fig2")
plt.grid(True)
plt.plot(train_errors,marker='o',label='training set polynomial')
plt.plot(val_errors,marker='o',label='validation set polynomial')
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend()
plt.tight_layout()
plt.legend(framealpha=1)
plt.ylim(0,8)


#%% Ridge Regression 16
plt.figure("fig1")

poly_features = PolynomialFeatures(degree=16,include_bias=(False))
X_poly=poly_features.fit_transform(X)

rig = Ridge()
rig.fit(X_poly,y)

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = rig.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='g', label='Ridge[16]', linestyle='--')

plt.legend()


#%% Ridge Regression MSE
model = Ridge()

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

model = Pipeline((
    ('poly_features', PolynomialFeatures(degree=16, include_bias=False)),
    ('ridge_reg', Ridge(alpha=1)),
    ))

train_errors, val_errors = [], []
for i in range(1,len(X_train)):
    model.fit(X_train[:i],y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    
    #compute the error for the trained model
    mse_train = mean_absolute_error(y_train[:i],y_train_predict)
    train_errors.append(mse_train)
    
    #compute the error for the validation model
    mse_val = mean_absolute_error(y_val, y_val_predict)
    val_errors.append(mse_val)
    
    plt.figure('test model')
    plt.scatter(X, y, s=2, label='data')
    plt.scatter(X_train[:1], y_train[:1], s=2, label='data in training set')
    plt.scatter(X_val[:1], y_val[:1], marker='s', label='validation set')
    y_model=model.predict(X_model)
    plt.plot(X_model,y_model,'r--',label='model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('test_plots/linear_model'+str(i))
    plt.close('test model')
    
plt.figure("fig2")
plt.grid(True)
plt.plot(train_errors,marker='s',linestyle='--',label='training set ridge')
plt.plot(val_errors,marker='s',linestyle='--',label='validation set ridge')
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend()
plt.tight_layout()
plt.legend(framealpha=1)
plt.ylim(0,8)