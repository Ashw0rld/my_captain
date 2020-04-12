# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:01:02 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\KIIT\Documents\Python Scripts\mnist_test.csv')

a = df.iloc[18][1:].values

a = a.reshape(28,28).astype('uint8')

plt.imshow(a)

X = np.array(df.drop(['label'], 1))

Y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=4)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train, y_train)

accuracy = rf.score(x_test, y_test)
print(accuracy)