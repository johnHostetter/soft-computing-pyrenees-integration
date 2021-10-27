#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:14:51 2021

@author: john
"""

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from functools import partial

from clip import CLIP
from fis import gaussian
from lazypop import noise_criterion
from constant import PROBLEM_FEATURES
from policy_induction_problem import initial_target

# return the dataset as sample of traces: <student, s, a, r, done>
def getTrace(filename, feature_list):
    raw_data = pd.read_csv(filename)
    feature_len = len(feature_list)
    trace = []
    student_list = list(raw_data['userID'].unique())
    
    for student in student_list:
        student_data = raw_data.loc[raw_data['userID'] == student,]
        row_index = student_data.index.tolist()

        for i in range(0, len(row_index)):

            state1 = student_data.loc[row_index[i], feature_list].values
            action_type = student_data.loc[row_index[i], 'action']

            if action_type == 'problem':
                action = 0
            elif action_type == 'example':
                action = 2
            else:
                action = 1

            reward = student_data.loc[row_index[i], 'inferred_rew'] * 10

            Done = False
            if (i == len(row_index) - 1):  # the last row is terminal state.
                Done = True
                state2 = np.zeros(feature_len)
            else:
                state2 = student_data.loc[row_index[i+1], feature_list].values

            state1 = np.asarray(state1).astype(np.float64)
            state2 = np.asarray(state2).astype(np.float64)
            trace.append([state1, action, reward, state2, Done])

    return trace, feature_len

def undo_normalization(data_path, problem_id):
    # load training data set X and Y
    print('loading normalized data...')
    # normalized_data = pd.read_csv('labeled_critical_train_data.csv', delimiter=',')
    normalized_data = pd.read_csv(data_path, delimiter=',')
    # undo normalization before train/val/test split
    
    print('undoing normalization ...')
    # problem_id = 'problem'
    normalization_vector_path = './normalization_values/normalization_features_all_{}.csv'.format(problem_id)
    df = pd.read_csv(normalization_vector_path)
    min_vector = df.min_val.values.astype(np.float64)
    max_vector = df.max_val.values.astype(np.float64)
    raw_data = deepcopy(normalized_data)
    
    for feature in df.feat.values:
        row = df[df['feat'] == feature]
        max_val = row['max_val'].values[0]
        min_val = row['min_val'].values[0]
        raw_data[feature] = (raw_data[feature] * (max_val - min_val)) + min_val
    
    print('normalization undone...')
    return raw_data

def cluster_membership(X, membership_functions):
    membership_degree = 1.0
    for x, term in zip(X, membership_functions):
        membership_degree = min(membership_degree, term(x))
        # membership_degree *= term(x)
    return membership_degree

def make_cluster_membership_functions(membership_functions):
    # cluster_membership_functions = []
    clusters = []
    for cluster_membership_function in itertools.product(*membership_functions):
        cluster_membership_functions = []
        for membership_function in cluster_membership_function:
            cluster_membership_functions.append(partial(gaussian, center=membership_function['center'], sigma=membership_function['sigma']))
        clusters.append(partial(cluster_membership, membership_functions=cluster_membership_functions))
    return clusters

def rule_creation(X, antecedents):
    # Wang and Mendel approach to fuzzy logic rule creation, but without the consequent portion
    start = time.time()
    rules = []
    weights = []
    for idx, x in enumerate(X):    
        if idx % 500 == 0:
            print('idx: %s' % idx)
        CF = 1.0 # certainty factor of this rule
        A_star_js = []
        for p in range(len(x)):
            SM_jps = []
            for j, A_jp in enumerate(antecedents[p]):
                SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                SM_jps.append(SM_jp)
            CF *= np.max(SM_jps)
            j_star_p = np.argmax(SM_jps)
            A_star_js.append(j_star_p)
            
        R_star = {'A':A_star_js, 'CF': CF, 'time_added': start}
        
        if not rules:
            # no rules in knowledge base yet
            rules.append(R_star)
            weights.append(1.0)
        else:
            # check for uniqueness
            add_new_rule = True
            for k, rule in enumerate(rules):
                if (rule['A'] == R_star['A']):
                    # the generated rule is not unique, it already exists, enhance this rule's weight
                    weights[k] += 1.0
                    rule['CF'] = min(rule['CF'], R_star['CF'])
                    add_new_rule = False
                    break
            if add_new_rule:
                rules.append(R_star)
                weights.append(1.0)
    return rules, weights

file_name = 'features_all_prob_action_immediate_reward'
data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)

# # feature selection using feature importance via RandomForestRegressor
# raw_data = pd.read_csv(data_path)
raw_data = undo_normalization(data_path, 'problem')

# # https://stackoverflow.com/questions/49282049/remove-strongly-correlated-columns-from-dataframe
# def trimm_correlated(df_in, threshold):
#     df_corr = df_in.corr(method='pearson', min_periods=1)
#     df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
#     un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
#     df_out = df_in[un_corr_idx]
#     return df_out

# trimmed_df = trimm_correlated(raw_data, 0.9)

trimmed_df = raw_data
AVAILABLE_PROBLEM_FEATURES = list(set(PROBLEM_FEATURES).intersection(set(trimmed_df.columns)))
X = trimmed_df[AVAILABLE_PROBLEM_FEATURES].values
Y = trimmed_df['inferred_rew'].values

# # https://pub.towardsai.net/pca-clearly-explained-when-why-how-to-use-it-and-feature-importance-a-guide-in-python-56b3da72d9d1
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Z-score the features
# scaler = StandardScaler()
# scaler.fit(X)
# standardized_X = scaler.transform(X)
# pca = PCA(.90) # estimate only 2 PCs
# X_new = pca.fit_transform(standardized_X)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())

# # find the most important features
# print(abs( pca.components_ ))

# weighted_sum_of_importances = (pca.explained_variance_ratio_[:, np.newaxis] * abs(pca.components_)).sum(axis=0)
# mean_importances = abs(pca.components_).mean(axis=0)

# # get the top 10 features
# # https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
# n = 10
# selected_features = []
# indices = (-weighted_sum_of_importances).argsort()[:n]
# for idx in indices:
#     print(AVAILABLE_PROBLEM_FEATURES[idx])
#     selected_features.append(AVAILABLE_PROBLEM_FEATURES[idx])

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
# model.fit(X, Y)
# # get importance
# importance = model.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#  	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()
# # get the indices of the n largest elements from a Numpy array
# n = 12
# ind = np.argpartition(importance, -n)[-n:]
# selected_features = np.array(PROBLEM_FEATURES)[ind]
# selected_features = PROBLEM_FEATURES[:4]


print('getting traces...')
selected_features = PROBLEM_FEATURES
traces, feature_len = getTrace(data_path, selected_features)
print('done.\n')

print('formatting data...')
student_state = []
for state1, action, reward, state2, done in traces:
    student_state.append(state1)
student_state = np.asarray(student_state)
print('done.\n')

X = student_state
X_reduced = X

# from sklearn.decomposition import PCA
# # pca = PCA(n_components=10)
# pca = PCA(.90)
# principalComponents = pca.fit_transform(X)
# X_reduced = principalComponents

# feature agglomeration and dimensionality reduction
# from sklearn import cluster
# agglo = cluster.FeatureAgglomeration(n_clusters=30)
# agglo.fit(X)
# X_reduced = agglo.transform(X)

print('generating antecedents...')
antecedents = CLIP(X_reduced, X_reduced, 
                   X_reduced.min(axis=0), X_reduced.max(axis=0), 
                   terms=[], alpha=0.5, beta=0.7, theta=0.0)
print('done.\n')

rules, weights = rule_creation(X_reduced, antecedents)

# keep only the rules that were generated by more than one data observation
rule_indices = np.where(np.array(weights) > 3)[0]
selected_rules = list(np.array(rules)[rule_indices])
selected_weights = list(np.array(weights)[rule_indices])

from nfqn import NeuroFuzzyQNetwork

neuro_fuzzy = NeuroFuzzyQNetwork()

# print('making fuzzy clusters...')
# input_space_clusters = make_cluster_membership_functions(antecedents)
# print('input space done (%s identified)...' % len(input_space_clusters))
# # output_space_clusters = make_cluster_membership_functions(consequents)
# # print('output space done (%s identified)...' % len(output_space_clusters))

# # --- INPUT SPACE ---

# medians = []
# for input_cluster in input_space_clusters:
#     xs = []
#     for x in X:
#         xs.append(input_cluster(x))
#     medians.append(np.median(xs))
#     plt.scatter(range(len(X)), sorted(xs, reverse=True), s=0.75, alpha=0.75)
# input_NC = np.mean(medians)
# plt.hlines(input_NC, 0, len(X), colors='grey', linestyles='dashed', label='Noise Criterion')
# plt.xlabel('Observations')
# plt.ylabel('Minimum Degree of Membership to Input Space Clusters')
# plt.title('Identification of Scrupulous Data in the Input Space Domain')
# plt.legend()
# plt.show()

# noise_determination = noise_criterion()