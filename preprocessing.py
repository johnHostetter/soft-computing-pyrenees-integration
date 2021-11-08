#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:45:00 2021

@author: john
"""

import numpy as np
import pandas as pd

from copy import deepcopy
from itertools import compress

from constant import PROBLEM_FEATURES, STEP_FEATURES

def undo_normalization(data_path, policy_type):
    """
    Undoes the normalization that may have already been applied to the provided data.

    Parameters
    ----------
    data_path : string
        The filepath where the data is saved.
    policy_type : string
        The type of policy that is being induced; options are 'problem' or any other string.
        If any string besides 'problem' is passed as an argument, the encode_action function
        will encode the 'action_type' argument as it would if it were for a step-level policy.
        This is done because the 'policy_type' argument when inducing a step-level policy, is
        the problem ID that the step-level policy is meant to be used on (e.g. 'exc137' or 'ex132a').

    Returns
    -------
    raw_data : Pandas DataFrame
        The Pandas DataFrame representation of the .csv file located at 'data_path',
        after the normalization has been undone.

    """
    # load training data set X and Y
    print('loading normalized data...')
    normalized_data = pd.read_csv(data_path, delimiter=',')
    print('done.\n')
    print('undoing normalization ...')
    normalization_vector_path = './normalization_values/normalization_features_all_{}.csv'.format(policy_type)
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
    return raw_data, max_vector, min_vector

def encode_action(policy_type, action_type):
    """
    Encode the provided 'action_type' string into its corresponding integer representation,
    dependent on what type of policy is being induced (i.e. 'policy_type').

    Parameters
    ----------
    policy_type : string
        The type of policy that is being induced; options are 'problem' or any other string.
        If any string besides 'problem' is passed as an argument, the encode_action function
        will encode the 'action_type' argument as it would if it were for a step-level policy.
        This is done because the 'policy_type' argument when inducing a step-level policy, is
        the problem ID that the step-level policy is meant to be used on (e.g. 'exc137' or 'ex132a').
    action_type : string
        The action taken, as recorded in the .csv file.

    Returns
    -------
    action : int
        The integer representation of the action taken.

        For problem-level policy:
            case 0: problem-solving (PS)
            case 1: faded-worked-example (FWE)
            case 2: worked-example (WE)

        For step-level policy:
            case 0: elicit
            case 1: tell

    """
    if policy_type == 'problem': # problem-level actions
        if action_type == 'problem':
            action = 0
        elif action_type == 'example':
            action = 2
        else:
            action = 1
    else: # step-level actions
        if action_type == 'problem':
            action = 0
        else:
            action = 1
    return action

def policy_features(policy_type, filter=None):
    """
    Depending on the policy that is being induced, retrieve the features that are
    relevant to the decision-level.

    Parameters
    ----------
    policy_type : string
        The type of policy that is being induced; options are 'problem' or any other string.
        If any string besides 'problem' is passed as an argument, the encode_action function
        will encode the 'action_type' argument as it would if it were for a step-level policy.
        This is done because the 'policy_type' argument when inducing a step-level policy, is
        the problem ID that the step-level policy is meant to be used on (e.g. 'exc137' or 'ex132a').
    filter : list, optional
        The 'filter' argument is a list of booleans, where 'True' at the i'th index indicates that
        the i'th feature should be kept for this 'policy_type'. If no 'filter' argument is provided,
        then no features are selected. The default is None.

    Returns
    -------
    list
        A list of string elements, where each element is a feature of the data.

    """
    if policy_type == 'problem':
        if filter is None:
            return PROBLEM_FEATURES
        else:
            return list(compress(PROBLEM_FEATURES, filter))
    else:
        if filter is None:
            return STEP_FEATURES
        else:
            return list(compress(STEP_FEATURES, filter))

def inferred_reward_constant(policy_type):
    """
    Depending on the policy that is being induced, the inferred reward from InferNet
    is multiplied by a different constant. This was done in Song's original
    Critical HRL policy induction.

    Parameters
    ----------
    policy_type : string
        The type of policy that is being induced; options are 'problem' or any other string.
        If any string besides 'problem' is passed as an argument, the encode_action function
        will encode the 'action_type' argument as it would if it were for a step-level policy.
        This is done because the 'policy_type' argument when inducing a step-level policy, is
        the problem ID that the step-level policy is meant to be used on (e.g. 'exc137' or 'ex132a').

    Returns
    -------
    int
        A constant that the inferred reward is multipled by, which is dependent on the type of policy being induced.

    """
    if policy_type == 'problem':
        return 10
    else:
        return 100

# return the dataset as sample of traces: <student, s, a, r, done>
def build_traces(filename, policy_type, filter=None):
    """
    Return the dataset as a list of traces in the format of:

        (state, action, reward, next state, done)

    Parameters
    ----------
    filename : string
        The filepath where the data is saved.
    policy_type : string
        The type of policy that is being induced; options are 'problem' or any other string.
        If any string besides 'problem' is passed as an argument, the encode_action function
        will encode the 'action_type' argument as it would if it were for a step-level policy.
        This is done because the 'policy_type' argument when inducing a step-level policy, is
        the problem ID that the step-level policy is meant to be used on (e.g. 'exc137' or 'ex132a').
    filter : list, optional
        The 'filter' argument is a list of booleans, where 'True' at the i'th index indicates that
        the i'th feature should be kept for this 'policy_type'. If no 'filter' argument is provided,
        then no features are selected. The default is None.

    Returns
    -------
    traces : list
        A list containing elements in the form of:
            (state, action, reward, next state, done)
    number_of_features : int
        The number of [problem/step] features.

    """
    raw_data = pd.read_csv(filename)
    feature_list = policy_features(policy_type, filter)
    number_of_features = len(feature_list)
    traces = []
    student_list = list(raw_data['userID'].unique())

    for student in student_list:
        student_data = raw_data.loc[raw_data['userID'] == student,]
        row_index = student_data.index.tolist()

        for i in range(0, len(row_index)):
            state1 = student_data.loc[row_index[i], feature_list].values
            action_type = student_data.loc[row_index[i], 'action']
            action = encode_action(policy_type, action_type)
            reward = student_data.loc[row_index[i], 'inferred_rew'] * inferred_reward_constant(policy_type)
            done = False

            if (i == len(row_index) - 1):  # the last row is terminal state.
                done = True
                state2 = np.zeros(number_of_features)
            else:
                state2 = student_data.loc[row_index[i+1], feature_list].values

            state1 = np.asarray(state1).astype(np.float64)
            state2 = np.asarray(state2).astype(np.float64)
            traces.append([state1, action, reward, state2, done])

    return traces, number_of_features
