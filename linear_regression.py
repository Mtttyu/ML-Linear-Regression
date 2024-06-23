# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd



path = 'data.txt'
data = pd.read_csv(path, header=None, names=['one','two','three'])

cols = data.shape[1]
X=data.iloc[:,0:cols-2]
y=data.iloc[:,cols-2:cols-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=23)

model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_train)


print(X_train.shape)
print(y_train.shape)

plt.scatter(X_train,y_train)
plt.plot(X_train,y_predict,color='r')

acc=r2_score(y_train, y_predict)
print(acc)




