#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 02:34:49 2021

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
from constant import PROBLEM_LIST, STEP_FEATURES
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
            else:
                action = 1

            reward = student_data.loc[row_index[i], 'inferred_rew'] * 100

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

num_actions = 2
for problem in PROBLEM_LIST:
    policy_type = problem
    # policy_type = 'ex252'
    file_name = 'features_all_{}'.format(policy_type)
    data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)
    
    try:
        raw_data = undo_normalization(data_path, policy_type)
    except FileNotFoundError: # this problem has no pedagogical agent intervention
        continue
    
    print('getting traces...')
    selected_features = STEP_FEATURES
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
    
    columns = ['elicit_Q_value', 'tell_Q_value']
    for idx in range(num_actions):
        column_name = columns[idx]
        raw_data[column_name] = predicted_q_values[:, idx]
        
    raw_data.to_csv('./policy_output/{}_q_values.csv'.format(policy_type), sep=',')