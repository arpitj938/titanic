# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 23:17:36 2017

@author: arpit
"""

import numpy as np
import matplotlib as plt
import pandas as pd
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print test.head()

#print train.head()
print train.columns[train.isnull().any()]

numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])

corr = numeric_data.corr()
#sns.heatmap(corr)

#fare will consider as a parameter
print corr['Survived'].sort_values(ascending=False)[:]

#cat = [f for f in train.columns if train.dtypes[f] == 'object']
#
#def anova(frame):
#    anv = pd.DataFrame()
#    anv['features'] = cat
#    pvals = []
#    for c in cat:
#           samples = []
#           for cls in frame[c].unique():
#                  s = frame[frame[c] == cls]['Survived'].values
#                  samples.append(s)
#           pval = stats.f_oneway(*samples)[1]
#           pvals.append(pval)
#    anv['pval'] = pvals
#    return anv.sort_values('pval')
#
#cat_data['Survived'] = train.Survived.values
#k = anova(cat_data) 
#k['disparity'] = np.log(1./k['pval'].values) 
#sns.barplot(data=k, x = 'features', y='disparity') 
#plt.xticks(rotation=90) 
#plt 

#sex and ticket will be parameter

#num = [f for f in train.columns if train.dtypes[f] != 'object']
#num.remove('PassengerId')
#nd = pd.melt(train, value_vars = num)
#n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
#n1 = n1.map(sns.distplot, 'value')
#n1

#filling missing data
train.drop(train[train['Fare'] > 300].index, inplace=True)
train['Age'].fillna(int(stats.mode(train['Age']).mode),inplace=True)
test['Age'].fillna(int(stats.mode(train['Age']).mode),inplace=True)
fill_na = str(stats.mode(train['Cabin']).mode)
train['Cabin'].fillna(fill_na,inplace=True)
test['Cabin'].fillna(fill_na,inplace=True)
fill_na = str(stats.mode(train['Embarked']).mode)
train['Embarked'].fillna(fill_na,inplace=True)
fill_na = int(stats.mode(train['Fare']).mode)
test['Fare'].fillna(fill_na,inplace=True)

#print test.columns[test.isnull().any()]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def factorize(data, var, fill_na = None):
      if fill_na is not None:
            data[var].fillna(fill_na, inplace=True)
      le.fit(data[var])
      data[var] = le.transform(data[var])
      return data

#combine the data set
alldata = train.append(test)
print alldata.shape

for x in cat_data.columns:
    factorize(alldata,x)



train_new = alldata[alldata['Survived'].notnull()]
test_new = alldata[alldata['Survived'].isnull()]
print train_new.head()
label_df = pd.DataFrame(index = train_new.index, columns = ['Survived'])
label_df['Survived'] = train_new['Survived']
print 'label',label_df.head()   
print label_df['Survived'].isnull().count()

import xgboost as xgb

regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=10,
                       min_child_weight=4,
                       n_estimators=17200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1,
                       objective='binary:logistic'
                       )                      
#bst = xgb.train(params,dtrain,nround)
regr.fit(train_new, label_df)
#
from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

# run prediction on training set to get an idea of how well it does
y_pred = []
for f in regr.predict(train_new):
    if f>=0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_test = label_df
print y_pred[:5]
print y_test[:5]
print("XGBoost score on training set: ", rmse(y_test, y_pred))

## make prediction on test set
y_pred_xgb = []
print regr.predict(test_new)[:10]
for f in regr.predict(test_new):
    if f>=0.05:
        y_pred_xgb.append(1)
    else:
        y_pred_xgb.append(0)
print y_pred_xgb[:5]
#submit this prediction and get the score
pred1 = pd.DataFrame({'PassengerId': test_new['PassengerId'], 'Survived': y_pred_xgb})
pred1.to_csv('xgbnono.csv', header=True, index=False)
