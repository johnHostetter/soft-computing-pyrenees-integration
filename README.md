# Integration of the Soft Computing Library to Pyrenees
Repository Name: <strong>pyrenees_soft_integration</strong>

### Setup procedure
<p>This project allows for the integration between the Pyrenees code, and the Soft Computing library that has been developed by Hostetter. It essentially acts as a liaison between the two repositories, to simplify the utilization of soft computing algorithms in Pyrenees experiments.</p>

<p>To use this project code, a few steps must be followed during the installation. <strong>These steps are assuming that the <code>Pyrenees-Python</code> project has absolutely no Git submodule or copy of this project, or this project's dependencies (e.g. the <code>soft_computing</code> submodule).</strong></p>

<p>First, add this GitHub repository (<code>pyrenees_soft_integration</code>) as a Git submodule to the Pyrenees GitHub repository (<code>Pyrenees-python</code> at the time of writing). Specifically, it should be added in the following path:<code>Pyrenees-python/app/</code></p>

<p>To add this GitHub repository as a Git submodule, while in a terminal at the above path, type:
<code>git submodule add [remote url]</code>
where [remote url] is this GitHub repository's [HTTPS link](https://github.ncsu.edu/jwhostet/pyrenees_soft_integration.git).</p>

<p>There should now be a directory located at: <code>Pyrenees-python/app/pyrenees_soft_integration</code>. However, the soft computing algorithms are not yet ready to be used. Specifically, you will encounter import errors from the files contained within the <code>soft_computing</code> submodule (e.g. the files in the <code>fuzzy</code> directory).</p>

<p>To finish the setup and integration process, we need to do consult the [setup procedure](https://github.ncsu.edu/jwhostet/soft_computing) outlined in the <code>soft_computing</code> GitHub repository. However, the most relevant instructions are included here. Particularly, we need to create a Python virtual environment, or activate an existing Python virtual environment. Then, from within the <code>soft_computing</code> directory, we follow the pip installation steps, but only need to run:<br>
<code>pip install -e .</code>

<p>The code should now be ready to use. We can easily import any code from this GitHub repository (<code>pyrenees_soft_integration</code>), and its subdirectories, into <code>Pyrenees-python/app/routes.py</code> (which is where pedagogical decision making occurs), by following the same convention that is used for libraries such as Numpy or Pandas (e.g. <code>import numpy as np</code>).</p>

### Explanation of scripts and files
The following scripts are of primary interest:

  <code>problem_level_policy_induction.py</code>
  <code>step_level_policy_induction.py</code>

<p>The first script will induce a problem-level policy for Pyrenees. The second script will induce a separate step-level policy for each problem ID in Pyrenees.</p>

<p>The <code>constant.py</code> script file contains a few constants of interest. Specifically, only the following are used:</p>

  <code>PROBLEM_LIST</code>  
  <code>PROBLEM_FEATURES</code>  
  <code>STEP_FEATURES</code>

<p>The <code>PROBLEM_LIST</code> is a list that contains the problem IDs stored as strings. Some problem IDs have no pedagogical intervention, and as such, there will be no decision information on these, as every student receives the same intervention (e.g. the first "problem" is always worked-example for every student). As such, there are some try-catch blocks in the code meant to handle this scenario when the provided problem ID has no corresponding decision information.</p>

<p>The <code>PROBLEM_FEATURES</code> is a list that contains the features stored as strings, that are only available for problem-level decisions. These values correspond directly to the column names of the provided <code>.csv</code> files containing the problem-level/step-level decision information.</p>

<p>Similarly, <code>STEP_FEATURES</code> is a list that contains the features stored as strings, that are only available for step-level decisions. <code>PROBLEM_FEATURES</code> is not the same list as <code>STEP_FEATURES</code>. Specifically, <code>PROBLEM_FEATURES</code> has fewer features (only 130 features, where <code>STEP_FEATURES</code> is 142 features).</p>

<p>The other constants found in the script file such as <code>MEDIAN_THRESHOLD_LTR</code> or <code>MEDIAN_THRESHOLD_STR_POSITIVE</code> are legacy constants that are used by Song's Critical HRL. They are kept as a precautionary measure in case they are needed once more.</p>

<p>The <code>preprocessing.py</code> script file contains a few functions of interest. Specifically, the following:</p>

  <code>undo_normalization()</code><br>
  <code>encode_action()</code><br>
  <code>policy_features()</code><br>
  <code>inferred_reward_constant()</code><br>
  <code>build_traces()</code><br>

<p>The <code>undo_normalization()</code> function will take the provided data, and reverse any normalization that has been applied to it. However, it does not check that the data has been normalized. This function exists since the original training data provided by Song was normalized, and in a prior study, the normalization had to be undone.</p>

<p>The <code>encode_action()</code> function will take the policy type that is being induced (e.g. problem-level or step-level), and encode the action as it was saved originally (in a string format) as an integer. Refer to the documentation for more details.</p>

<p>The <code>policy_features()</code> function is a simple function that exists in order to allow the <code>build_traces</code> function to be independent of the policy type. This <code>policy_features()</code> function simply returns the features required for the provided policy type.</p>

<p>The <code>inferred_reward_constant()</code> function is a simple function that exists in order to allow the <code>build_traces()</code> function to be independent of the policy type. This <code>inferred_reward_constant()</code> function simply returns the constant required for the provided policy type. Specifically, the constant returned is multiplied by the inferred immediate reward at this decision point.</p>

<p>The <code>build_traces()</code> function returns the provided data as a list where each element has the form:</p>

  <code>(state, action, reward, next state, done)</code>

<p>This representation is required for the reinforcement learning algorithms to be applied.</p>

<p>The <code>soft_computing</code> directory is a [Git submodule](https://git-scm.com/docs/git-submodule). To add a Git submodule (at the time of writing), use the following command:

  <code>git submodule add [remote url]</code>

where <code>[remote url]</code> is the URL to your remote GitHub repository.</p>

<p>Git submodules have their HEAD pointer frozen to when they were added, so they will not be automatically be updated if the remote repository receives updates. To update the Git submodule(s), the submodule(s) can be updated in a testing branch with the following command:

  <code>git submodule update</code>

If nothing is negatively affected, the changes/fixes can then be merged to the main branch.</p>

<p>For the original post advocating for Git submodules, see [here](https://stackoverflow.com/questions/45557791/suggestion-on-import-python-module-from-another-github-project).</p>
