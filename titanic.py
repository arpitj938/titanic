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
cat_data = train.select_dtypes(exclude=[np.number])

corr = numeric_data.corr()
#sns.heatmap(corr)

#fare will consider as a parameter
print corr['Survived'].sort_values(ascending=False)[:]

cat = [f for f in train.columns if train.dtypes[f] == 'object']

def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['Survived'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cat_data['Survived'] = train.SalePrice.values
k = anova(cat_data) 
k['disparity'] = np.log(1./k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt 