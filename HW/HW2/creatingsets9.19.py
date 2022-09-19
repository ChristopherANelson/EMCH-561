# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 23:30:41 2022

@author: Owner
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:30:41 2022

@author: Christopher Nelson, Alex

HW-2 - EMCH 567
"""


import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", 
                              values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
   
plt.close('all')

#%% Load and plot data
oecd_bli = pd.read_csv("data\oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("data\gdp_per_capita.csv",thousands=',',
                             delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
GDP = country_stats.values[:,0]
life_satisfaction = country_stats.values[:,1]
countries = list(country_stats.index)

#%% Plot original data
'''
plt.figure()
plt.xlabel('GDP')
plt.ylabel('Life Satisfaction')
plt.title('GDP vs Life Satisfaction')
plt.plot(GDP,life_satisfaction,'o')
'''

#%% Normalization
data = np.asarray(GDP.reshape(-1,1))
xMax=max(data)
xMin=min(data)
yMax=max(life_satisfaction)
yMin=min(life_satisfaction)

X=np.asarray([(xMax-i)/(xMax-xMin) for i in data])
Y=np.asarray([(yMax-i)/(yMax-yMin) for i in life_satisfaction])


#%% Closed Form
X_model = np.linspace(xMin-4000,xMax+4000)
Y=np.expand_dims(Y, axis=1)

X_b = np.ones((X.shape[0],2))
X_b[:,1]=X.T

thetaCF=np.linalg.inv(X_b.T@X_b)@X_b.T@Y

scaled_thetaCF=[yMax-(yMax-yMin)*(thetaCF[0]+thetaCF[1]*(xMax/(xMax-xMin))),(yMax-yMin)/(xMax-xMin)*thetaCF[1]]
life_satisfaction_modelCF=scaled_thetaCF[0]+scaled_thetaCF[1]*X_model


#%% Gradient Descent
def Gradient_Descent(eta,n_iterations,X_b,Y):
    n=X_b.shape[0]
    np.random.seed(117)
    theta=np.random.rand(2,1)
    theta_list = np.array(theta)
    for iteration in range(n_iterations):
        gradients = 2/n*X_b.T@(X_b@theta-Y)
        theta=theta-eta*gradients
        theta_list=np.append(theta_list,theta,axis=1)
    return theta, theta_list

def Scale(dataX, dataY, modelCoeff):
    xMax=max(dataX)
    xMin=min(dataX)
    yMax=max(dataY)
    yMin=min(dataY)
    return [yMax-(yMax-yMin)*(modelCoeff[0]+modelCoeff[1]*(xMax/(xMax-xMin))),(yMax-yMin)/(xMax-xMin)*modelCoeff[1]]
    
theta_gd,theta_list_gd = Gradient_Descent(0.3, 120, X_b, Y)

scaled_theta_gd=[yMax-(yMax-yMin)*(theta_gd[0]+theta_gd[1]*(xMax/(xMax-xMin))),(yMax-yMin)/(xMax-xMin)*theta_gd[1]]

life_satisfaction_model_gd=scaled_theta_gd[0]+scaled_theta_gd[1]*X_model

#Low eta and low n_iterations Set
eta_low = 0.003
iter_low = 10
theta_lowei,ugh1 = Gradient_Descent(eta_low, iter_low, X_b, Y)
#mid eta and mid n_iter
eta_mid = 0.03
iter_mid = 50
theta_midei, ugh2 = Gradient_Descent(eta_mid, iter_mid, X_b, Y)
#high eta and high n_iter
eta_high = 0.3
iter_high = 120
theta_highei, ugh3 = Gradient_Descent(eta_high, iter_high, X_b, Y)

theta_low_ei = Scale(X_model, life_satisfaction, theta_lowei)
grad_model_low_ei = theta_low_ei[0] + theta_low_ei[1] * X_model

theta_mid_ei = Scale(X_model, life_satisfaction, theta_midei)
grad_model_mid_ei = theta_mid_ei[0] + theta_mid_ei[1] * X_model


theta_highei = Scale(X_model, life_satisfaction, theta_highei)
grad_model_high_ei = theta_highei[0] + theta_highei[1] * X_model

# theta_low_ei,theta_list_ei = Gradient_Descent(0.0005, 25, X_b, Y)

# scaled_theta_low_ei = Scale(data, life_satisfaction, theta_low_ei)

# gd_model_low_ei = scaled_theta_low_ei[0] + scaled_theta_low_ei[1] * X_model
#%% Plotting

#Figure 1

plt.figure(figsize=(6.5,3))
plt.plot(X_model,life_satisfaction_modelCF, '--', color='k', label = 'Closed Form Model')

plt.plot(X_model,life_satisfaction_model_gd, '--', color='r', label = 'Gradient Descent Model')
plt.plot(data, life_satisfaction, 'o', markersize=7, label='training data')

plt.xlabel('GDP (U.S. Dollars)')
plt.ylabel('OCED Life Satsifaction Index')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Figure 2
plt.figure(figsize=(6.5,3))
plt.plot(data, life_satisfaction, 'o', markersize=7, label='training data')

labellow = f"Eta: {eta_low}, Iteration Number: {iter_low}"
plt.plot(X_model, grad_model_low_ei, '^', label = labellow)

labelmid = f"Eta: {eta_mid}, Iteration Number: {iter_mid}"
plt.plot(X_model, grad_model_mid_ei, '<', label = labelmid)

labelhigh = f"Eta: {eta_high}, Iteration Number: {iter_high}"
plt.plot(X_model, grad_model_high_ei, '>', label = labelhigh)

plt.plot(X_model,life_satisfaction_model_gd, '--', color='r', label = 'Gradient Descent Model')


plt.grid(True)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(6.5,3))
eta_increm = 0.050
iter_increm = 30
eta  = 0.01
itern = 10
for iteration in range(0, 10):
    theta, superf = Gradient_Descent(eta, itern, X_b, Y)
    thetaScaled = Scale(X_model, life_satisfaction, theta)
    model = thetaScaled[0] + thetaScaled[1] * X_model
    
    labels = f"Eta: {eta}, Iteration Number: {itern}"
    plt.plot(X_model, model, label = labels)
    eta += eta_increm
    itern += iter_increm 
#plt.plot(X_model,life_satisfaction_model_gd, '--', color='r', label = 'Gradient Descent Model')
plt.plot(X_model,life_satisfaction_modelCF, '*', color='k', label = 'Closed Form Model')

plt.grid(True)
plt.legend()
plt.tight_layout()












