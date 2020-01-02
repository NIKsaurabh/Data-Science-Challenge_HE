#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:32:09 2020

@author: saurabh
"""

import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dummy = pd.get_dummies(train['Result'])

train = pd.concat([train,dummy], axis = 1)

train.drop(['ID','Result'],axis = 1,inplace = True)
X = train.iloc[:,:-6].values
y = train.iloc[:,-6:].values

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)

model.fit(X,y)

test.drop('ID',axis=1,inplace = True)
predict = model.predict(test)

predict = pd.DataFrame(predict, columns=['h0','h1','h2','h3','h4','h5'])

x = predict.stack()

pre = (pd.Categorical(x[x!=0].index.get_level_values(1)))

pre = pd.DataFrame([pre],columns=[])

pre.to_csv('submission.csv',index = False)
