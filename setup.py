#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:23:56 2021

@author: john
"""

import os
import sys
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path

from fuzzy.reinforcement.cfql import CFQLModel

GLOBAL_MODEL = None
SYS_PATH_TO_THIS_FILE_PARENT = str(Path(__file__).parent.absolute())
sys.path.append(SYS_PATH_TO_THIS_FILE_PARENT)

try:  # meant for testing the connection between this library of code on the server
    from app.pyrenees_soft_integration.constant import PROBLEM_LIST, STEP_FEATURES
except ModuleNotFoundError:  # meant for testing locally, like on your PC
    from constant import PROBLEM_LIST, STEP_FEATURES

class FuzzyPedagogicalAgent(object):
    def __init__(self):
        self.lookup_table = {}
        self.min_vectors = {}
        self.max_vectors = {}
        self.build()

    def build(self):
        problem_ids = ['problem']
        problem_ids.extend(PROBLEM_LIST)

        # the below parameters are not necessarily needed (except the fis_params' inference_engine)
        # but they are included because in the current implementation, they are still expected to be there
        clip_params = {'alpha':0.1, 'beta':0.7}
        fis_params = {'inference_engine':'product'}

        # for each problem ID, build its corresponding model, and save it in the dictionaries
        for problem_id in problem_ids:
            local_path_to_model = '/models/{}/{}'.format(problem_id, problem_id)
            model_file_name = SYS_PATH_TO_THIS_FILE_PARENT + local_path_to_model
            normalization_vector_path = SYS_PATH_TO_THIS_FILE_PARENT + '/normalization_values/normalization_features_all_{}.csv'.format(problem_id)
            if problem_id == 'problem':
                action_set_length = 3
            else:
                action_set_length = 2

            # the below parameters are not necessarily needed (except the fis_params' inference_engine)
            # but they are included because in the current implementation, they are still expected to be there
            # note this alpha for CQL is different than CLIP's alpha
            cql_params = {
                'gamma':0.0, 'alpha':0.0, 'batch_size':0, 'batches':0,
                'learning_rate':0.0, 'iterations':0 ,'action_set_length':action_set_length
                }
            try:
                # abstract problem ID type
                self.lookup_table[problem_id] = CFQLModel(clip_params, fis_params, cql_params)
                self.lookup_table[problem_id].load(model_file_name)
                normalization_df = pd.read_csv(normalization_vector_path)
                self.min_vectors[problem_id] = normalization_df.min_val.values.astype(np.float64)
                self.max_vectors[problem_id] = normalization_df.max_val.values.astype(np.float64)

                # word problem ID type
                self.lookup_table[problem_id+'w'] = CFQLModel(clip_params, fis_params, cql_params)
                self.lookup_table[problem_id+'w'].load(model_file_name)
                normalization_df = pd.read_csv(normalization_vector_path)
                self.min_vectors[problem_id+'w'] = normalization_df.min_val.values.astype(np.float64)
                self.max_vectors[problem_id+'w'] = normalization_df.max_val.values.astype(np.float64)
            except FileNotFoundError:
                continue # this is assumed to be a problem_id where there is no associated pedagogical decision making (e.g. ex222)

    def predict(self, decision_level, problem_id, z, normalized=False):
        # z will be unnormalized, raw data observation, meant for true integration with Pyrenees
        if not normalized:
            norm_z = (z - self.min_vectors[problem_id]) / (self.max_vectors[problem_id] - self.min_vectors[problem_id])
        else:
            norm_z = z
        model = self.lookup_table[problem_id]

        # select only the features that are applicable for this policy
        filter = (self.max_vectors[problem_id] != self.min_vectors[problem_id])
        norm_z = np.take(norm_z, np.where(filter)[0], axis=0)

        action = model.get_action(norm_z)
        return action

def load_model():
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        GLOBAL_MODEL = FuzzyPedagogicalAgent()
    return GLOBAL_MODEL

def logic_mapping(decision_level, problem_id, raw_prediction):
    if decision_level == 'problem':
        return 'problem' if raw_prediction == 0 else 'step_decision' if raw_prediction == 1 else 'example'
    else:
        return 'problem' if raw_prediction == 0 else 'example'

def fuzzy_decision(decision_level, problem_id, input_features):
    decision_info = {'decision': 'step_decision', 'Qvalues': '-1', 'policy': 'fcql'}
    #try:
    if True:
        model = load_model()
        if len(input_features) == 130 or len(input_features) == 142:
            raw_prediction = model.predict(decision_level, problem_id, np.array(input_features), normalized=False)
            decision = logic_mapping(decision_level, problem_id, raw_prediction)
            decision_info['decision'] = decision
            decision_info['Qvalues'] = str(raw_prediction)
            return decision_info
        else:
            decision_info['Qvalues'] = '-2' # default to FWE with Qvalues as -2 to represent length of z was violated
            return decision_info
    #except Exception:
        return decision_info # default to FWE with Qvalues value as -1 to represent exception thrown
