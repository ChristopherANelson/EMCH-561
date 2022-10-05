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
plt.xlabel("X Data")
plt.ylabel("Y Data")

plt.figure("fig3")
plt.scatter(X_train,y_train, label='Training Set')
plt.scatter(X_val,y_val, label='Validation Set', marker='X')
plt.ylim(0,10)
plt.xlabel("X Data")
plt.ylabel("Y Data")

plt.figure("fig5")
plt.scatter(X_train,y_train, label='Training Set')
plt.scatter(X_val,y_val, label='Validation Set', marker='X')
plt.ylim(0,10)
plt.legend()
plt.xlabel("X Data")
plt.ylabel("Y Data")

#%% POLY REGRESSION 2
plt.figure("fig1")
poly_features = PolynomialFeatures(degree=2,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='tab:orange', label='Polynomial [2]')


#%% POLY REGRESSION 20
poly_features = PolynomialFeatures(degree=20,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = reg.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='purple', label='Polynomial [20]', linestyle='dashdot')


#%% Linear Regression MSE
model = LinearRegression()

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

model = Pipeline((
    ('poly_features', PolynomialFeatures(degree=20, include_bias=False)),
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
    
plt.figure("fig2")
plt.grid(True)
plt.plot(train_errors,marker='s',color='tab:blue',label='training set polynomial')
plt.plot(val_errors,marker='s',linestyle='--',color='g',label='validation set polynomial')
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend()
plt.tight_layout()
plt.legend(framealpha=1)
plt.ylim(0,8)


#%% Ridge Regression 20
plt.figure("fig1")

poly_features = PolynomialFeatures(degree=20,include_bias=(False))
X_poly=poly_features.fit_transform(X)

rig = Ridge(alpha=1)
rig.fit(X_poly,y)

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals = rig.predict(X_vals_poly)

plt.plot(X_vals, y_vals, color='g', linewidth = 2,label='Ridge[20]', linestyle='--')

plt.legend()


#%% Ridge Regression MSE
model = Ridge()

X_model=np.linspace(-3,3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

model = Pipeline((
    ('poly_features', PolynomialFeatures(degree=20, include_bias=False)),
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
    
plt.figure("fig2")
plt.grid(True)
plt.plot(train_errors,marker='o',color='tab:orange',label='training set ridge')
plt.plot(val_errors,marker='o',linestyle='--',color = 'tab:red', label='validation set ridge')
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend()
plt.tight_layout()
plt.legend(framealpha=1)
plt.ylim(0,8)

#%% generate alpha plots
plt.figure("fig3")
alphas = [0,0.1,1,10]

poly_features = PolynomialFeatures(degree=20,include_bias=(False))
X_poly=poly_features.fit_transform(X)

linestyles = ['-','--','dashdot','dotted']
colors = ['g','r','orange','purple']

for i in range(0,4):
    rig = Ridge(alpha=alphas[i])
    rig.fit(X_poly,y)

    X_model=np.linspace(-3,3,100).reshape(-1,1)
    X_vals_poly=poly_features.transform(X_vals)

    y_vals = rig.predict(X_vals_poly)

    labels = "Ridge[20], alpha = {}".format(alphas[i])
    plt.plot(X_vals, y_vals, linestyle = linestyles[i], c=colors[i], label = labels, markevery = 5)

plt.legend()

#%% generate learning curves for alpha plots
for j in range(0,4):
    model = Ridge()
    
    X_model=np.linspace(-3,3,100).reshape(-1,1)
    X_vals_poly=poly_features.transform(X_vals)
    
    model = Pipeline((
        ('poly_features', PolynomialFeatures(degree=20, include_bias=False)),
        ('ridge_reg', Ridge(alpha=alphas[j])),
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
        
    error = round(val_errors[-1]-train_errors[-1],4)
    plt.figure()
    plt.grid(True)
    plt.plot(train_errors,marker='o',color='tab:orange',label='training set ridge, alpha = {}'.format(alphas[j]))
    plt.plot(val_errors,marker='s',linestyle='--',color = 'tab:red', label='validation set ridge, alpha = {}'.format(alphas[j]))
    plt.plot([],[],' ',label = "Final Error = {}".format(error))
    plt.xlabel('number of data points in training set')
    plt.ylabel('mean squared error')
    plt.legend(framealpha=1)
    plt.ylim(0,8)
    

