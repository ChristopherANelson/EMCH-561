# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 23:30:41 2022

@author: Owner
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PrepareCountryStats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", 
                              values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, 
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    remove_labels = full_country_stats.index[remove_indices]
    full_country_stats.drop(remove_labels, axis=0, inplace=True)
    return full_country_stats[["GDP per capita", 'Life satisfaction']]
   
plt.close('all')

#%% Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', 
                             delimiter='\t', encoding='latin1', 
                             na_values="n/a")

#%% Prepare the data
country_stats = PrepareCountryStats(oecd_bli, gdp_per_capita)
GDP = country_stats.values[:,0]
life_satisfaction = country_stats.values[:,1]

#%% Normalization
xMax = max(GDP)
xMin = min(GDP)
yMax = max(life_satisfaction)
yMin = min(life_satisfaction)
xRange = xMax-xMin
yRange = yMax-yMin

X = (xMax-GDP)/(xRange)
Y = (yMax-life_satisfaction)/(yRange)

#%% Closed Form
X_model = np.linspace(xMin-4000,xMax+4000)
Y = np.expand_dims(Y, axis=1)

X_b = np.ones((X.shape[0],2))
X_b[:,1] = X.T

thetaCF = np.linalg.inv(X_b.T@X_b)@X_b.T@Y

scaled_thetaCF = np.empty((2,1))
scaled_thetaCF[0] = yMax-(yRange)*(thetaCF[0]+thetaCF[1]*(xMax/(xRange)))
scaled_thetaCF[1] = (yRange)/(xRange)*thetaCF[1]
life_satisfaction_modelCF = scaled_thetaCF[0]+scaled_thetaCF[1]*X_model

#%% Gradient Descent
def GradientDescent(eta,n_iterations,X_b,Y):
    n=X_b.shape[0]
    np.random.seed(117)
    theta=np.random.rand(2,1)
    theta_list = np.array(theta)
    for iteration in range(n_iterations):
        gradients = 2/n*X_b.T@(X_b@theta-Y)
        theta=theta-eta*gradients
        theta_list=np.append(theta_list,theta,axis=1)
    return theta, theta_list

#%% Scaling
def Scale(dataX, dataY, modelCoeff):
    xMax=max(dataX)
    xMin=min(dataX)
    yMax=max(dataY)
    yMin=min(dataY)
    xRange = xMax-xMin
    yRange = yMax-yMin
    
    thetaScaled = np.empty((2,1))
    thetaScaled[0] = yMax-(yRange)*(modelCoeff[0]+modelCoeff[1]*(xMax/xRange))
    thetaScaled[1] = (yRange)/(xRange)*modelCoeff[1]
    return thetaScaled
    
#%% Plot the figures
plt.figure(figsize=(6.5,3))
plt.plot(GDP, life_satisfaction, 'o', markersize=7, label='training data')
eta_increm = 0.1
iter_increm = 60
eta  = 0.01
itern = 10
markers = [None,'D', 's', '*']
colors = ['g','r','orange','purple']

# Calculate gradient descent lines for various input parameters
for i in range(0,4):
    eta_val = round(eta + eta_increm * i, 2)
    iter_val = round(itern + iter_increm * i, 2)
    theta, superf = GradientDescent(eta_val, iter_val, X_b, Y)
    thetaScaled = Scale(X_model, life_satisfaction, theta)
    model = thetaScaled[0] + thetaScaled[1] * X_model
    theta1 = round(thetaScaled[0][0], 2)
    theta2 = round(thetaScaled[1][0], 8)
    
    labels = f"Eta: {eta_val}, Iteration Number: {iter_val}, Theta: ({theta1}, {theta2})"
    plt.plot(X_model, model, marker=markers[i], c=colors[i],label = labels)

#Plot closed form model
plt.plot(X_model,life_satisfaction_modelCF, '--', color='k', 
         label = 'Closed Form Model')
plt.xlabel("GDP (USD)")
plt.ylabel('OCED Life Satisfaction Index')

plt.grid(True)
plt.legend()
plt.tight_layout()












