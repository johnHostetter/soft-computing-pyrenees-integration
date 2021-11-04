# pyrenees_fall_2021_study

The following scripts are of primary interest:

  problem_level_policy_induction.py  
  step_level_policy_induction.py  

The first script will induce a problem-level policy for Pyrenees.
The second script will induce a separate step-level policy for each problem ID in Pyrenees.

The 'constant.py' script file contains a few constants of interest. Specifically, only the following are used:

  PROBLEM_LIST  
  PROBLEM_FEATURES  
  STEP_FEATURES  

The 'PROBLEM_LIST' is a list that contains the problem IDs stored as strings. Some problem IDs have no pedagogical intervention,
and as such, there will be no decision information on these, as every student receives the same intervention (e.g. the first "problem" is always worked-example for every student). As such, there are some try-catch blocks in the code meant to handle this scenario when the provided problem ID has no corresponding decision information.

The 'PROBLEM_FEATURES' is a list that contains the features stored as strings, that are only available for problem-level decisions. These values correspond directly to the column names of the provided .csv files containing the problem-level/step-level decision information.

Similarly, 'STEP_FEATURES' is a list that contains the features stored as strings, that are only available for step-level decisions. 'PROBLEM_FEATURES' is not the same list as 'STEP_FEATURES'. Specifically, 'PROBLEM_FEATURES' has fewer features (only 130 features, where 'STEP_FEATURES' is 142 features).

The other constants found in the script file such as 'MEDIAN_THRESHOLD_LTR' or 'MEDIAN_THRESHOLD_STR_POSITIVE' are legacy constants that are used by Song's Critical HRL. They are kept as a precautionary measure in case they are needed once more.

The 'preprocessing.py' script file contains a few functions of interest. Specifically, the following:

  undo_normalization  
  encode_action  
  policy_features  
  inferred_reward_constant  
  build_traces  

The 'undo_normalization' function will take the provided data, and reverse any normalization that has been applied to it. However, it does not check that the data has been normalized. This function exists since the original training data provided by Song was normalized, and in a prior study, the normalization had to be undone.

The 'encode_action' function will take the policy type that is being induced (e.g. problem-level or step-level), and encode the action as it was saved originally (in a string format) as an integer. Refer to the documentation for more details.

The 'policy_features' function is a simple function that exists in order to allow the 'build_traces' function to be independent of the policy type. This 'policy_features' function simply returns the features required for the provided policy type.

The 'inferred_reward_constant' function is a simple function that exists in order to allow the 'build_traces' function to be independent of the policy type. This 'inferred_reward_constant' function simply returns the constant required for the provided policy type. Specifically, the constant returned is multiplied by the inferred immediate reward at this decision point.

The 'build_traces' function returns the provided data as a list where each element has the form:

  (state, action, reward, next state, done)  

This representation is required for the reinforcement learning algorithms to be applied.

The 'fuzzy' directory is a [Git submodule](https://git-scm.com/docs/git-submodule). To add a Git submodule (at the time of writing), use the following command:

  git submodule add [remote url]  

where [remote url] is the URL to your remote Git/GitHub repository.

Git submodules have their HEAD pointer frozen to when they were added, so they will not be automatically be updated if the remote repository receives updates. To update the Git submodule(s), the submodule(s) can be updated in a testing branch with the following command:

  git submodule update  

If nothing is negatively affected, the changes/fixes can then be merged to the main branch.

For the original post advocating for Git submodules, see [here](https://stackoverflow.com/questions/45557791/suggestion-on-import-python-module-from-another-github-project).
