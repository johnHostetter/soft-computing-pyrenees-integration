#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:14:51 2021

@author: john
"""

import os
import time
import torch
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from functools import partial

from clip import CLIP
from fis import gaussian
from constant import PROBLEM_FEATURES
from policy_induction_problem import initial_target

SEED = 0
os.environ['PYTHONHASHSEED']=str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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

num_actions = 3
policy_type = 'problem'
file_name = 'features_all_prob_action_immediate_reward'
data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)

raw_data = undo_normalization(data_path, policy_type)

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

from cfql import CFQLModel

cfql = CFQLModel(alpha=0.6, beta=0.7, gamma=0.99, learning_rate=1e-2, ee_rate=0., action_set_length=num_actions)
cfql.fit(X, traces, ecm=True, Dthr=0.125, prune_rules=False, apfrb_sensitivity_analysis=False,)
print('q-table consequents')
print(np.unique(np.argmax(cfql.q_table, axis=1), return_counts=True))

actions = []
predicted_q_values = []
for x in X:
    q_values = cfql.infer(x).tolist()
    action = np.argmax(q_values)
    predicted_q_values.append(q_values)
    actions.append(action)
    
actions = np.array(actions)
predicted_q_values = np.array(predicted_q_values)

print('actual distribution')
print(np.unique(actions, return_counts=True))

print('original distribution')
original_actions = [trace[1] for trace in traces]
print(np.unique(original_actions, return_counts=True))

print('%.2f%% similarity.' % (100 * np.count_nonzero(np.array(actions) == np.array(original_actions)) / X.shape[0]))

columns = ['ps_Q_value', 'fwe_Q_value', 'we_Q_value']
for idx in range(num_actions):
    column_name = columns[idx]
    raw_data[column_name] = predicted_q_values[:, idx]
    
raw_data.to_csv('./policy_output/{}_q_values.csv'.format(policy_type), sep=',')

# storage = None
# batch_size = 100
# n_batches = int(np.round(X.shape[0]/batch_size))
# for idx in range(n_batches):
#     print('batch %s' % idx)
#     batch_X = X[idx*batch_size:(idx+1)*batch_size]
#     o1 = cfql.input_layer(batch_X)
#     o2 = cfql.condition_layer(o1)
#     o3 = cfql.rule_base_layer(o2, inference='product')
#     if storage is None:
#         storage = deepcopy(o3)
#     else:
#         storage = np.vstack((storage, deepcopy(o3)))
    
# max_rule_activations = np.mean(storage, axis=0)
# # plt.bar([x for x in range(len(mean_rule_activations))], mean_rule_activations)
# plt.plot(range(len(max_rule_activations)), sorted(max_rule_activations))
# plt.show()

# certainty_and_weights = np.array([(cfql.rules[idx]['CF'], cfql.weights[idx]) for idx in range(cfql.K)])
# plt.scatter(certainty_and_weights[:,0], (certainty_and_weights[:,1] / X.shape[0]), s=0.5)
# plt.show()

# outlier1 = sorted(certainty_and_weights[:,0])[-1]
# outlier2 = sorted(certainty_and_weights[:,0])[-2]

# outlier1_idx = np.where(certainty_and_weights[:,0] == outlier1)[0]
# certainty_and_weights = np.delete(certainty_and_weights, outlier1_idx, axis=0)
# outlier2_idx = np.where(certainty_and_weights[:,0] == outlier2)[0]
# certainty_and_weights = np.delete(certainty_and_weights, outlier2_idx, axis=0)
# plt.scatter(certainty_and_weights[:,0], (certainty_and_weights[:,1] / X.shape[0]), s=0.5)
# plt.show()

# print('generating antecedents...')
# antecedents = CLIP(X, X_reduced, 
#                    X_reduced.min(axis=0), X_reduced.max(axis=0), 
#                    terms=[], alpha=0.5, beta=0.7, theta=0.0)
# print('done.\n')

# rules, weights = rule_creation(X_reduced, antecedents)

# # keep only the rules that were generated by more than one data observation
# rule_indices = np.where(np.array(weights) > 3)[0]
# selected_rules = list(np.array(rules)[rule_indices])
# selected_weights = list(np.array(weights)[rule_indices])

# from nfqn import NeuroFuzzyQNetwork

# neuro_fuzzy = NeuroFuzzyQNetwork()

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