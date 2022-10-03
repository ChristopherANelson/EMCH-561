# -*- coding: utf-8 -*-
"""


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

#%% Plot generate

X= 4* np.random.rand(100,1)-2
y = 4+2*X+5*X**2+np.random.rand(100,1)


poly_features = PolynomialFeatures(degree=2,include_bias=(False))
X_poly=poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly,y)

X_vals=np.linspace(-2,2,100).reshape(-1,1)
X_vals_poly=poly_features.transform(X_vals)




y_vals = reg.predict(X_vals_poly)



plt.plot(X_vals, y_vals, color='r')

plt.scatter(X,y)
plt.show()