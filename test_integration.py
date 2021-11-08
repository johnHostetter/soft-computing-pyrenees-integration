import sys
import pandas as pd

from pathlib import Path

"""
    This script is meant to be run locally on the desktop,
    to check integration with Pyrenees prior to server deployment.

    Essentially, it can be used to verify the decision_info dictionary
    will be created correctly when the pedagogical agent is called.
"""

SYS_PATH_TO_THIS_FILE_PARENT = str(Path(__file__).parent.absolute())
sys.path.append(SYS_PATH_TO_THIS_FILE_PARENT + '/soft_computing/')

from setup import fuzzy_decision
from preprocessing import undo_normalization, build_traces
from constant import PROBLEM_LIST, PROBLEM_FEATURES, STEP_FEATURES

policies = ['problem']
policies.extend(PROBLEM_LIST)

for policy in policies:
    policy_type = policy
    if policy_type == 'problem':
        file_name = 'features_all_prob_action_immediate_reward'
        data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)
    else:
        file_name = 'features_all_{}'.format(policy_type)
        data_path = 'training_data/nn_inferred_{}.csv'.format(file_name)

    try:
        raw_data, _, _ = undo_normalization(data_path, policy_type)
    except FileNotFoundError:
        continue # this problem has no decision making

    for i in range(len(raw_data)):
        if policy_type == 'problem':
            info = fuzzy_decision('problem', policy_type, raw_data.iloc[i][PROBLEM_FEATURES])
        else:
            info = fuzzy_decision('step', policy_type, raw_data.iloc[i][STEP_FEATURES])

        print(policy_type + ' ' + str(i) + ' ' + str(info))
