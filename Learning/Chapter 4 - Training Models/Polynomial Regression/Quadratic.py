import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import linear_model

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

m = 100
X = 6*np.random.rand(m,1)-3
y = 0.5 * X**2 + X +2 +np.random.randn(m,1)

plt.plot(X,y,'b.')

'''
Clearly, a straight line will never fit this data properly.
Let's use Scikit-Learn's PolynomialFeatures class to transform our training data, adding the square (second-degree polynomial) of each feature in the training set as a new feature (in this case there is just one feature):
'''

from sklearn.preprocessing import PolynomialFeatures

poly_features=PolynomialFeatures(degree=2, include_bias=False)
X_poly=poly_features.fit_transform(X)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)

lin_reg.predict(X_poly)