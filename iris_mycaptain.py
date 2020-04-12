# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:58:07 2020

@author: KIIT
"""

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

df = pd.read_csv(r'C:\Users\KIIT\Documents\Python Scripts\iris.csv')

df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

df.hist()

scatter_matrix(df)

x = np.array(df.drop(['variety'],1))
y = np.array(df['variety'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#building models
models = []
models.append(('lr',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('lda',LinearDiscriminantAnalysis()))
models.append(('knn',KNeighborsClassifier()))
models.append(('nb',GaussianNB()))
models.append(('svm',SVC()))

resultss = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    resultss.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
#LinearDiscriminantAnalysis was found to be most efficient.
    
plt.boxplot(resultss, labels=names)
plt.title('Algorithm Comaprison')
plt.show()

model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)
#evaluate our prediction
print(accuracy)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))











