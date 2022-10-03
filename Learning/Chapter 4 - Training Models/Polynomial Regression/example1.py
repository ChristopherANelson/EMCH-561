
'''
EXAMPLE 1
'''
import IPython as IP
IP.get_ipython().magic('reset -sf')

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn import datasets, preprocessing, linear_model, pipeline
from sklearn.linear_model import LinearRegression



plt.close('all')

#%% PLOT GENERATION
m=100
X=6*np.random.rand(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)



#%% Regression

from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg=LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-3, 3,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)

y_vals=reg.predict(X_vals_poly)


Coef=reg.coef_.T
intercept=reg.intercept_



#%% PLOTTING
plt.figure()
plt.scatter(X,y,color='b')

pred_label=str(Coef[1]) +'X^2 + ' + str(Coef[0]) + 'X + ' + str(intercept)
plt.plot(X_vals,y_vals,color='r',label='predictions')
plt.grid(True)
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.show()

#%% PRINT FUNCTION


#print('Function is predicted as ' + str(Coef[1]) +'X^2 + ' str(Coef[0]) +'X +' + str(intercept))

print(pred_label)

