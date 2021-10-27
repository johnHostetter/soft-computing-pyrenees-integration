# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:26:58 2021

@author: jhost
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

EPSILON = 5e-3 # any feature with importance greater than epsilon is kept

def feature_importance_by_extra_trees(X, y):
    tree = ExtraTreesClassifier(n_estimators=10)
    tree.fit(X, y)
    return tree.feature_importances_

def filter_by_extra_trees_epsilon(X, y, epsilon=EPSILON):
    # determine the importance of features
    important_features = feature_importance_by_extra_trees(X, y)
    
    # plot the feature importance
    plt.plot(range(len(important_features)), sorted(important_features))
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance by Extra Trees Classifier')
    plt.show()

    # select only the most important features from all the features
    return select_by_important_features(X, important_features, epsilon)

def select_by_important_features(X, important_features, epsilon=EPSILON):
    indices = np.where(important_features > epsilon)[0]
    feature_selected_state = np.take(X, indices, axis=-1)
    return feature_selected_state.astype('float64'), len(indices), important_features

def explained_variance_by_pca(X, y):
    n_components = len(X[0])
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    print(np.cumsum(pca.explained_variance_ratio_))

    # plot the explained variance
    plt.plot(range(n_components), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Feature Importance by Extra Trees Classifier')
    plt.show()
    
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
    
def correlation(df, n):
    print("Correlation Matrix")
    print(df.corr())
    print()    
    print("Top Absolute Correlations")
    print(get_top_abs_correlations(df, n))