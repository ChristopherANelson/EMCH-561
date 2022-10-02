'''
MNIST dataset

70,000 small images of digits handritten by high school students and employees of US Census Bureau
Often called the 'hello world' of Machine Learning

'''

#%% Importing Code
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

import numpy as np
import scipy as sp



'''
Datasets have:
    A DESCR key describing the dataset
    A data key containing an array with one row per instance and one columb per feature
    a target key containing an array with labels
'''



X=np.asarray(mnist['data'])
y=np.asarray(mnist['target'],dtype='int')



'''
>>> X.shape
(70000,784)

>>> y.shape
(70000)

70,000 images, each having 784 features.
28 x 28 pixels = 784
'''

#%% Let's take features and reshape it to a 28x28 array and display it using Matplotlib using imshow()

import matplotlib as mpl
import matplotlib.pyplot as plt


some_digit=X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis('off')
plt.show()





#%% Splitting into training and testing
'''
Splitting into a training set and test set
    training set will be first 60,000 images
    test set will be last 10,000 images
'''

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#%% Training a Binary Classifier
# Identify the number 5
'''
Binary Classifier is capable of distinguishing between only two classes, 5 and not-5
'''
y_train_5 = (y_train==5) #true for all 5s, false for all other digits
y_test_5 = (y_test==5)

'''
Picking a classifier
Pick Stochastic Gradient Descent (SGD)
    has advantage of being capable of handling very large datasets efficiently
    deals with training instances independently one at a time
'''

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
#SGD relies on randomness during training. If you want reporducible results, you should set the random_state parameter
sgd_clf.fit(X_train,y_train_5)

print(sgd_clf.predict([some_digit])) # guesses that this image represents a 5
























