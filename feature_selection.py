# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:14:07 2021

@author: jhost
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

from constant import PROBLEM_FEATURES
warnings.filterwarnings("ignore")

np.random.seed(123)

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns

def feature_selection(original_df, KEY):
    df = pd.read_csv('./data/strat_train_data.csv')
    features_list = deepcopy(PROBLEM_FEATURES)
    features_list.append(KEY)
    df = df[features_list]
    
    if len(PROBLEM_FEATURES) > 131:
        df.drop(columns=df.columns[-1], axis=1, inplace=True)
    
    label_encoder = LabelEncoder()
    df[KEY] = label_encoder.fit_transform(df[KEY]).astype('float64')
    
    corr = df.corr()
    sns.heatmap(corr)
    
    # drop any feature that has a linear correlation of 0.9 or greater
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    
    selected_without_target = selected_columns.drop(KEY)
    
    selected_columns = selected_without_target
    
    SL = 0.05
    data_modeled, selected_columns = backwardElimination(df.drop(KEY, axis=1).values, df[KEY].values, SL, selected_columns)
    
    new_df = pd.DataFrame(data_modeled, columns=selected_columns)
    new_df['label'] = original_df['label']
    new_df['critical'] = original_df['critical']
    if False:
        new_df.to_csv('./data/{}_feature_selected_strat_train_data.csv'.format(KEY), index=False)
    return new_df

original_df = pd.read_csv('./data/strat_train_data.csv')
#feature_selection(original_df, 'critical')
feature_selection(original_df, 'label')