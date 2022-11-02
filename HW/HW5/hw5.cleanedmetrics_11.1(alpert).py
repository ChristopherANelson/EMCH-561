# -*- coding: utf-8 -*-
"""
EMCH 561 - HW5
Christopher Nelson | Name | Name | Name
"""



#%% Set-Up
import numpy as np
import sklearn as sk
from sklearn import datasets, linear_model, multiclass
from sklearn.svm import LinearSVC
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sklearn.metrics
from sklearn.metrics import classification_report


cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.close('all')


#%% Loading data
iris = sk.datasets.load_iris()
X = iris['data']
X_sepal = iris['data'][:,:2]
Y = iris['target']
Y_names = iris['target_names']
feature_names = iris['feature_names']


#%% Splitting data
X_train = X_sepal[0:40,:]
X_test = X_sepal[40:,:]
Y_train = Y[0:40]
Y_test = Y[40:]

#%% Softmax Regression 
softmax_regression = sk.linear_model.LogisticRegression(multi_class='multinomial', solver = 'lbfgs',C=10)
softmax_regression.fit(X_sepal, Y)

x0, x1 = np.meshgrid(np.linspace(0,9,500),np.linspace(0,5,200))
X_new = np.vstack((x0.reshape(-1),x1.reshape(-1))).T
y_proba = softmax_regression.predict_proba(X_new)
y_predict = softmax_regression.predict(X_new)
zz_predict = y_predict.reshape(x0.shape)
zz_proba = y_proba[:,1].reshape(x0.shape)

plt.figure(figsize=(7,5))
plt.xlim(3,9)
plt.ylim(1.5,5)
plt.contourf(x0,x1,zz_predict, cmap='Pastel2')

plt.grid(True)
plt.scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],color=cc[1])
plt.scatter(X[Y==2,0],X[Y==2,1],marker='d',label=Y_names[2],color=cc[2])
plt.scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],color=cc[0])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Softmax Regression")
plt.legend()
plt.tight_layout()

#%% linear model with One-Vs-Rest multiclass classifier

scaler = sk.preprocessing.MinMaxScaler()
X_sepal_norm = scaler.fit_transform(X_sepal)

ovr_clf = sk.multiclass.OneVsRestClassifier(sk.linear_model.LogisticRegression())
ovr_clf.fit(X_sepal_norm,Y)

x0, x1 = np.meshgrid(np.linspace(-2,2,5000),np.linspace(-2,2,2000))
X_new = np.vstack((x0.reshape(-1),x1.reshape(-1))).T
y_predict = ovr_clf.predict(X_new)
X_new = scaler.inverse_transform(X_new)
x0_ = np.reshape(X_new[:,0],(2000,5000))
x1_ = np.reshape(X_new[:,1],(2000,5000))

# convert back to matrix form
zz_predict = y_predict.reshape(x0.shape)

plt.figure(figsize=(7,5))
plt.xlim(3,9)
plt.ylim(1.5,5)
plt.contourf(x0_,x1_,zz_predict, cmap='Pastel2')
plt.grid(True)
plt.scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],color=cc[1])
plt.scatter(X[Y==2,0],X[Y==2,1],marker='d',label=Y_names[2],color=cc[2])
plt.scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],color=cc[0])
plt.title("a) One-verse-Rest - LOGISTIC REGRESSION")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.tight_layout()

#%% SVM classifier with One-Vs-Rest multiclass classifier
  
scaler = sk.preprocessing.MinMaxScaler()
X_sepal_norm = scaler.fit_transform(X_sepal.astype(np.float64))

svm_clf = sk.multiclass.OneVsRestClassifier(LinearSVC())
svm_clf.fit(X_sepal_norm, Y)

x0, x1 = np.meshgrid(np.linspace(-2,2,5000),np.linspace(-2,2,2000))
X_new = np.vstack((x0.reshape(-1),x1.reshape(-1))).T


y_predict = svm_clf.predict(X_new)
X_new = scaler.inverse_transform(X_new)
x0_ = np.reshape(X_new[:,0],(2000,5000))
x1_ = np.reshape(X_new[:,1],(2000,5000))

# convert back to matrix form
zz_predict = y_predict.reshape(x0.shape)
# zz_proba = y_proba[:,1].reshape(x0.shape)

plt.figure(figsize=(7,5))
plt.xlim(3,9)
plt.ylim(1.5,5)
plt.contourf(x0_,x1_,zz_predict, cmap='Pastel2')
# contour = plt.contour(x0, x1,zz_proba, [0.25,0.5,0.75], cmap='Pastel1')
# plt.clabel(contour, inline=1,fontsize=10)
plt.grid(True)
plt.scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],color=cc[1])
plt.scatter(X[Y==2,0],X[Y==2,1],marker='d',label=Y_names[2],color=cc[2])
plt.scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],color=cc[0])
plt.title("One-verse-Rest - SUPPORT VECTOR MACHINES")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.tight_layout()

#%% Model Metrics
#softmax regression
Y_train_pred = sk.model_selection.cross_val_predict(softmax_regression, X_sepal_norm, Y, cv=3)
sm_metrics=sk.metrics.classification_report(Y, Y_train_pred,target_names=Y_names, digits=4)
print("Softmax Regression Metrics\n" + sm_metrics)

#ovr model
Y_pred_ovr=sk.model_selection.cross_val_predict(ovr_clf, X_sepal_norm, Y, cv=3)
ovr_metrics=sk.metrics.classification_report(Y, Y_pred_ovr,target_names=Y_names, digits=4)
print("One vs Rest Logistic Regression Metrics\n" + ovr_metrics)

#support model
Y_pred_svm=sk.model_selection.cross_val_predict(svm_clf, X_sepal_norm, Y, cv=3)
svm_metrics=sk.metrics.classification_report(Y, Y_pred_svm,target_names=Y_names, digits=4)
print("Support Vector Model Metrics\n" + svm_metrics)

#%% Confusion Matrices

def iris_confmxplot(Y,Ypred,conf_mx, title):
    plt.figure(figsize=(6,6))
    pos = plt.imshow(conf_mx)
    cbar =plt.colorbar(pos)
    plt.title(title)
    cbar.set_label('Amount of Instances')
    plt.ylabel('Actual Plant')
    plt.xlabel('Predicted Plant')
    locs,labels=plt.xticks()
    plt.xticks(ticks=[0, 1, 2], labels = [Y_names[0], Y_names[1], Y_names[2]])
    plt.yticks(ticks=[0, 1, 2], labels = [Y_names[0], Y_names[1], Y_names[2]])
    plt.tight_layout()
    
# #Softmax Plotting
sm_cfm = sk.metrics.confusion_matrix(Y,Y_train_pred)
iris_confmxplot(Y, Y_train_pred, sm_cfm, "Softmax Confusion Matrix")

# ovr plotting
ovr_cfm = sk.metrics.confusion_matrix(Y,Y_pred_ovr)
iris_confmxplot(Y, Y_pred_ovr, ovr_cfm, "One vs. Rest Confusion Matrix")

#support vector
svm_cfm = sk.metrics.confusion_matrix(Y,Y_pred_svm)
iris_confmxplot(Y, Y_pred_svm, svm_cfm, "Support Vector Confusion Matrix")


# plt.figure(figsize=(6,6))
# pos = plt.imshow(sm_cfm)
# cbar =plt.colorbar(pos)
# plt.title("Softmax Classification Confusion Matrix")
# cbar.set_label('Amount of Instances')
# plt.ylabel('Actual Plant')
# plt.xlabel('Predicted Plant')
# locs,labels=plt.xticks()
# plt.xticks(ticks=[0, 1, 2], labels = [Y_names[0], Y_names[1], Y_names[2]])
# plt.yticks(ticks=[0, 1, 2], labels = [Y_names[0], Y_names[1], Y_names[2]])
# plt.tight_layout()

#sm_metrics=model_metrics(Y,Y_train_pred)


# # CM for Softmax
# Y_train_pred = sk.model_selection.cross_val_predict(softmax_regression, X_sepal_norm, Y, cv=3)
# confusion_matrix = sk.metrics.confusion_matrix(Y, Y_train_pred)
# print('Confusion Matrix: Softmax')
# print(confusion_matrix)

# SM_accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/confusion_matrix.sum()
# print('------Softmax Accuracy------')
# print('\n Accuracy = ' +str(SM_accuracy*100) +'%')


# SM_precision_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0])
# SM_precision_Versicolor=confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])
# SM_precision_Virginica=confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])

# print('\n ------Softmax Precision------- ')
# print('\n Setosa: ' +str(SM_precision_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_precision_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_precision_Virginica*100) +'%')


# SM_recall_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2])
# SM_recall_Versicolor=confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2])
# SM_recall_Virginica=confusion_matrix[2][2]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2])


# print('\n ------Softmax Recall------')
# print('\n Setosa: ' +str(SM_recall_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_recall_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_recall_Virginica*100) +'%')



# fig = plt.figure(figsize=(4,4))
# pos = plt.imshow(confusion_matrix)
# cbar =plt.colorbar(pos)
# plt.title("Softmax Classification Confusion Matrix")
# cbar.set_label('number of classified digits')
# plt.ylabel('actual digit')
# plt.xlabel('estimated digit')


# # CM for Linear Model
# Y_train_pred = sk.model_selection.cross_val_predict(ovr_clf, X_sepal_norm, Y, cv=3)
# confusion_matrix = sk.metrics.confusion_matrix(Y, Y_train_pred)
# print('\n Confusion Matrix: Linear Model')
# print(confusion_matrix)

# LOG_accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/confusion_matrix.sum()
# print('------Logistic OVR Accuracy------')
# print('\n Accuracy = ' +str(SM_accuracy*100) +'%')


# LOG_precision_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0])
# LOG_precision_Versicolor=confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])
# LOG_precision_Virginica=confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])

# print('\n ------Logistic OVR Precision------- ')
# print('\n Setosa: ' +str(SM_precision_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_precision_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_precision_Virginica*100) +'%')


# LOG_recall_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2])
# LOG_recall_Versicolor=confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2])
# LOG_recall_Virginica=confusion_matrix[2][2]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2])


# print('\n ------Logistic OVR Recall------')
# print('\n Setosa: ' +str(SM_recall_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_recall_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_recall_Virginica*100) +'%')




# fig = plt.figure(figsize=(4,4))
# pos = plt.imshow(confusion_matrix)
# cbar =plt.colorbar(pos)
# plt.title("Logistic OVR Confusion Matrix")
# cbar.set_label('number of classified digits')
# plt.ylabel('actual digit')
# plt.xlabel('estimated digit')

# # CM for Support Vector Machine
# Y_train_pred = sk.model_selection.cross_val_predict(svm_clf, X_sepal_norm, Y, cv=3)
# confusion_matrix = sk.metrics.confusion_matrix(Y, Y_train_pred)
# print('\n Confusion Matrix: Support Vector Machine')
# print(confusion_matrix)

# SVM_accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/confusion_matrix.sum()
# print('------SVM OVR Accuracy------')
# print('\n Accuracy = ' +str(SM_accuracy*100) +'%')


# SVM_precision_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0])
# SVM_precision_Versicolor=confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1])
# SVM_precision_Virginica=confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2])

# print('\n ------SVM OVR Precision------- ')
# print('\n Setosa: ' +str(SM_precision_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_precision_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_precision_Virginica*100) +'%')


# SVM_recall_Setosa=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[0][2])
# SVM_recall_Versicolor=confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2])
# SVM_recall_Virginica=confusion_matrix[2][2]/(confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2])


# print('\n ------SVM OVR Recall------')
# print('\n Setosa: ' +str(SM_recall_Setosa*100) +'%')
# print('\n Versicolor: ' +str(SM_recall_Versicolor*100) +'%')
# print('\n Virginica: '+str(SM_recall_Virginica*100) +'%')



# fig = plt.figure(figsize=(4,4))
# pos = plt.imshow(confusion_matrix)
# cbar =plt.colorbar(pos)
# plt.title("SVM OVR Confusion Matrix")
# cbar.set_label('number of classified digits')
# plt.ylabel('actual digit')
# plt.xlabel('estimated digit')


# confusion_matrix_noise = np.copy(np.asarray(confusion_matrix,dtype=np.float64))
# np.fill_diagonal(confusion_matrix_noise, np.NaN)

# fig = plt.figure(figsize=(4,4))
# pos = plt.imshow(confusion_matrix_noise)
# cbar =plt.colorbar(pos)
# cbar.set_label('number of classified digits')
# plt.ylabel('actual digit')
# plt.xlabel('estimated digit')

