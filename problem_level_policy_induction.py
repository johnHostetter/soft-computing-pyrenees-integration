#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:14:51 2021

@author: john
"""

import os
import sys
import torch
import random
import numpy as np

from preprocessing import undo_normalization, build_traces
from soft_computing.fuzzy.reinforcement.cfql import CFQLModel

""" --- Start of Reproducibility Code --- """

GLOBAL_SEED = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(GLOBAL_SEED)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(GLOBAL_SEED)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(GLOBAL_SEED)

# 4. Set the `torch` pseudo-random generator at a fixed value
torch.manual_seed(GLOBAL_SEED)

""" --- End of Reproducibility Code --- """

num_actions = 3
policy_type = 'problem'
file_name = 'features_all_prob_action_immediate_reward'
data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)

raw_data, _, _ = undo_normalization(data_path, policy_type)

print('getting traces...')
traces, feature_len = build_traces(data_path, policy_type)
print('done.\n')

print('formatting data...')
student_state = []
for state1, action, reward, state2, done in traces:
    student_state.append(state1)
student_state = np.asarray(student_state)
print('done.\n')

X = student_state

""" --- Start of Conservative Fuzzy Rule-Based Q-Learning Code --- """

clip_params = {'alpha':0.6, 'beta':0.7}
fis_params = {'inference_engine':'product'}
# note this alpha for CQL is different than CLIP's alpha
cql_params = {
    'gamma':0.99, 'alpha':0.1, 'batch_size':1028, 'batches':50,
    'learning_rate':1e-2, 'iterations':100 ,'action_set_length':num_actions
    }
cfql = CFQLModel(clip_params, fis_params, cql_params)
cfql.fit(X, traces, ecm=True, Dthr=0.125, prune_rules=False, apfrb_sensitivity_analysis=False, verbose=True)
try:
    cfql.save('./models/{}/{}'.format(policy_type, policy_type))
except FileNotFoundError:
    # the directory does not exist, so make the directory, then try again
    os.makedirs('./models/{}'.format(policy_type))
    cfql.save('./models/{}/{}'.format(policy_type, policy_type))

print('Q-table consequents')
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

""" --- End of Conservative Fuzzy Rule-Based Q-Learning Code --- """

print('distribution of estimated actions')
print(np.unique(actions, return_counts=True))
print()
print('distribution of original actions')
original_actions = [trace[1] for trace in traces]
print(np.unique(original_actions, return_counts=True))
print()
print('%.2f%% similarity.' % (100 * np.count_nonzero(np.array(actions) == np.array(original_actions)) / X.shape[0]))

columns = ['ps_Q_value', 'fwe_Q_value', 'we_Q_value']
for idx in range(num_actions):
    column_name = columns[idx]
    raw_data[column_name] = predicted_q_values[:, idx]

raw_data.to_csv('./policy_output/{}_q_values.csv'.format(policy_type), sep=',')
