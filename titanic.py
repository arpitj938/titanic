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

#print train.head()
#print train.columns[train.isnull().any()]

numeric_data = train.select_dtypes(include=[np.number])
print numeric_data.columns
cat_data = train.select_dtypes(exclude=[np.number])
print cat_data.columns

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

print test.columns[test.isnull().any()]
