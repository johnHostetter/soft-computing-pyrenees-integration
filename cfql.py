#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:56:26 2021

@author: john

    This code demonstrates the Conservative Fuzzy Rule-Based Q-Learning Algorithm.
        
    It is corrected from the FQL code at the following link: 
        https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning
    
    I have extended it with Tabular Conservative Q-Learning with code from:
        https://sites.google.com/view/offlinerltutorial-neurips2020/home
    
"""

import time
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from ecm import ECM
from nfn import AdaptiveNeuroFuzzy
from clip import CLIP, rule_creation

GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

def one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

class TabularNetwork(torch.nn.Module):
    def __init__(self, num_states, num_actions):
      super(TabularNetwork, self).__init__()
      self.num_states = num_states
      self.network = torch.nn.Sequential(
          torch.nn.Linear(self.num_states, num_actions)
      )

    def forward(self, states):
        onehot = one_hot(states, self.num_states)
        return self.network(onehot)

def get_tensors(list_of_tensors, list_of_indices):
    s, a, ns, r = [], [], [], []
    for idx in list_of_indices:
        s.append(list_of_tensors[idx][0])
        a.append(list_of_tensors[idx][1])
        r.append(list_of_tensors[idx][2])
        ns.append(list_of_tensors[idx][3])
    s = np.array(s)
    a = np.array(a)
    ns = np.array(ns)
    r = np.array(r)
    return s, a, ns, r

class CFQLModel(AdaptiveNeuroFuzzy):
    def __init__(self, alpha, beta, gamma, learning_rate, ee_rate, action_set_length):
        super().__init__()
        self.current_rule_activations = []
        self.R_ = []
        self.M = []
        self.V = []
        self.Q = []
        self.Error = 0
        self.alpha = alpha # the alpha parameter for the CLIP algorithm
        self.beta = beta # the beta parameter for the CLIP algorithm
        self.gamma = gamma # discount for future reward
        self.cql_alpha = 0.1 # the alpha parameter used in CQL
        self.learning_rate = learning_rate # the learning rate of CQL
        self.ee_rate = ee_rate # the exploration-exploitation rate, where a higher value indicates more exploration
        self.action_set_length = action_set_length
        
    def d(self, x):
        return self.truth_value(x).sum()
    
    def infer(self, x):
        self.truth_value(x)
        numerator = (self.current_rule_activations[:,np.newaxis] * self.q_table).sum(axis=0)
        denominator = self.current_rule_activations.sum()
        q_values = numerator / denominator
        return q_values
        
    def c_k(self, train_X, k):
        self.predict(train_X) # we just need the rule activations in layer 3, ignore the returned values
        lhs = 1 / self.f3.sum(axis=1)
        rhs = abs(self.q_table - self.q_table[k]).max()
        return lhs * rhs
        
    def get_number_of_rules(self):
        return self.K

    # Fuzzify to get the degree of truth values
    def truth_value(self, state_value):
        self.o1 = self.input_layer(state_value)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2, inference='product')
        self.current_rule_activations = copy.deepcopy(self.o3[0])
        return self.current_rule_activations

    def action_selection(self):
        self.M = []
        r = random.uniform(0, 1)

        for rule_index in range(self.get_number_of_rules()):
            # Act randomly
            if r < self.ee_rate:
                action_index = random.randint(0, self.action_set_length - 1)
            # Get maximum values
            else:
                state = np.array([rule_index])
                state_tensor = torch.tensor(state, dtype=torch.int64)
                q_values = self.network(state_tensor)
                q_values = q_values.detach().numpy() # detach from PyTorch
                action_index = np.argmax(q_values)
            self.M.append(action_index)

        action = self.M[np.argmax(self.current_rule_activations)]
        return action

    # Q(s,a) = Sum of (degree_of_truth_values[i]*q[i, a])
    def calculate_q_value(self):
        q_curr = 0
        for rule_index, truth_value in enumerate(self.current_rule_activations):
            state = np.array([rule_index])
            state_tensor = torch.tensor(state, dtype=torch.int64)
            q_values = self.network(state_tensor)
            q_values = q_values.detach().numpy() # detach from PyTorch
            q_curr += truth_value * q_values[0, self.M[rule_index]]
        self.Q.append(q_curr)

    # V'(s) = sum of (degree of truth values*max(q[i, a]))
    def calculate_state_value(self):
        # v_curr = 0
        # for index, rule in enumerate(self.q_table):
        #     v_curr += (self.current_rule_activations[index] * max(rule))
        # self.V.append(v_curr)
        v_curr = 0
        for rule_index in range(self.get_number_of_rules()):
            state = np.array([rule_index])
            state_tensor = torch.tensor(state, dtype=torch.int64)
            q_values = self.network(state_tensor)
            q_values = q_values.detach().numpy() # detach from PyTorch
            v_curr += (self.current_rule_activations[rule_index] * q_values.max())
        self.V.append(v_curr)

    # Q(i, a) += beta*degree_of_truth*delta_Q
    # delta_Q = reward + gamma*V'(s) - Q(s, a)
    def update_q_value(self, reward): # THIS STILL NEEDS UPDATED
        self.Error = reward + self.gamma * self.V[-1] - self.Q[-1]
        self.q_table = self.network(torch.arange(self.get_number_of_rules())).detach().numpy()
        # self.R_ is the degree of truth values for the previous state
        for index, truth_value in enumerate(self.R_):
            delta_q = self.learning_rate * (self.Error * truth_value)
            self.q_table[index][self.M[index]] += delta_q
        return self

    def save_state_history(self):
        self.R_ = copy.copy(self.current_rule_activations)

    def get_initial_action(self, state):
        self.V.clear()
        self.Q.clear()
        self.truth_value(state)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_action(self, state):
        q_values = self.infer(state)
        print(q_values)
        return np.argmax(q_values)

    def q_backup_sparse_sampled(self, q_values, state_index, action_index, 
                                next_state_index, reward, rule_weights, discount=0.99):
        next_state_q_values = q_values[next_state_index, :]
        values = np.max(next_state_q_values, axis=-1)
        target_value = (reward + discount * values)
        return target_value
    
    def project_qvalues_cql_sampled(self, state_index, action_index, target_values, 
                                    cql_alpha=0.1, num_steps=50, rule_weights=None):
        # train with a sampled dataset
        target_qvalues = torch.tensor(target_values, dtype=torch.float32)
        state_index = torch.tensor(state_index, dtype=torch.int64)
        action_index = torch.tensor(action_index, dtype=torch.int64)
        pred_qvalues = self.network(state_index)
        logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)
        
        pred_qvalues = pred_qvalues.gather(1, action_index.reshape(-1,1)).squeeze()
        cql_loss = logsumexp_qvalues - pred_qvalues

        loss = torch.mean((pred_qvalues - target_qvalues)**2)
        # loss = torch.mean(torch.tensor(rule_weights) * ((pred_qvalues - target_qvalues)**2))
        loss = loss + cql_alpha * torch.mean(cql_loss)
    
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        number_of_states = self.get_number_of_rules()
        pred_qvalues = self.network(torch.arange(number_of_states))
        return pred_qvalues.detach().numpy()
    
    def conservative_q_iteration(self, num_itrs=100, project_steps=50, cql_alpha=0.1, sampled=False,
                                 training_dataset=None, rule_weights=None, **kwargs):
        """
        Runs Conservative Q-iteration.
        
        Args:
          env: A GridEnv object.
          num_itrs (int): Number of FQI iterations to run.
          project_steps (int): Number of gradient steps used for projection.
          cql_alpha (float): Value of weight on the CQL coefficient.
          sampled (bool): Whether to use sampled datasets for training or not.
          training_dataset (list): list of (s, a, r, ns) pairs
          rule_weights (list): list of the rule activations where the i'th observation's current state (s) 
              correponds to the i'th element of rule_weights
        """
        
        number_of_states = self.get_number_of_rules()
        number_of_actions = self.action_set_length
        
        q_values = np.zeros((number_of_states, number_of_actions))
        for i in range(num_itrs):
            if sampled:
                for j in range(project_steps):
                    training_idx = np.random.choice(np.arange(len(training_dataset)), size=1028) # was 256
                    state_index, action_index, next_state, reward = get_tensors(training_dataset, training_idx)
                    
                    rule_weights_sample = np.array(rule_weights)[training_idx]
                    
                    target_values = self.q_backup_sparse_sampled(q_values, state_index, action_index, 
                                                                 next_state, reward, rule_weights_sample, **kwargs)
                    
                    intermediate_values = self.project_qvalues_cql_sampled(state_index, action_index, 
                                                                           target_values, cql_alpha=cql_alpha, rule_weights=rule_weights_sample)
                    if j == project_steps - 1:
                        q_values = intermediate_values
            else:
                raise Exception("The online version of Conservative Fuzzy Q-Learning is not yet available.")
        self.q_table = q_values
        return self.q_table
    
    def fit(self, train_X, trajectories, ecm=True, Dthr=1e-3, prune_rules=True, apfrb_sensitivity_analysis=False, verbose=True):
        print('The shape of the training data is: (%d, %d)' % (train_X.shape[0], train_X.shape[1]))
        train_X_mins = train_X.min(axis=0)
        train_X_maxes = train_X.max(axis=0)
        
        dummy_Y = np.zeros(train_X.shape[0])[:,np.newaxis] # this Y array only exists to make the rule generation simpler
        Y_mins = np.array([-1.0])
        Y_maxes = np.array([1.0])
        
        if verbose:
            print('Creating/updating the membership functions...')
        
        start = time.time()
        self.antecedents = CLIP(train_X, dummy_Y, train_X_mins, train_X_maxes, 
                                self.antecedents, alpha=self.alpha, beta=self.beta)
        end = time.time()
        start = time.time()
        if verbose:
            print('membership functions for the antecedents generated in %.2f seconds.' % (end - start))
        
        self.consequents = CLIP(dummy_Y, train_X, Y_mins, Y_maxes, self.consequents, 
                                alpha=self.alpha, beta=self.beta)
        end = time.time()
        if verbose:
            print('membership functions for the consequents generated in %.2f seconds.' % (end - start))
        
        if ecm:
            if verbose:
                print('Reducing the data observations to clusters using ECM...')
            start = time.time()
            clusters = ECM(train_X, [], Dthr)
            if verbose:
                print('%d clusters were found with ECM from %d observations...' % (len(clusters), train_X.shape[0]))
            reduced_X = [cluster.center for cluster in clusters]
            reduced_dummy_Y = dummy_Y[:len(reduced_X)]
            end = time.time()
            if verbose:
                print('done; the ECM algorithm completed in %.2f seconds.' % (end - start))
        else:
            reduced_X = train_X
            reduced_dummy_Y = dummy_Y
            
        if verbose:
            print('Creating/updating the fuzzy logic rules...')
        start = time.time()
        self.antecedents, self.consequents, self.rules, self.weights = rule_creation(reduced_X, reduced_dummy_Y,
                                                                                     self.antecedents,
                                                                                     self.consequents,
                                                                                     self.rules,
                                                                                     self.weights,
                                                                                     problem_type='SL',
                                                                                     consistency_check=False)
        
        K = len(self.rules)
        end = time.time()
        if verbose:
            print('%d fuzzy logic rules created/updated in %.2f seconds.' % (K, end - start))
            
        if verbose:
            print('Building the initial Neuro-Fuzzy Network...')
        start = time.time()
        self.import_existing(self.rules, self.weights, self.antecedents, self.consequents)
        self.orphaned_term_removal()
        self.preprocessing()
        self.update()
        end = time.time()
        if verbose:
            print('done; built in %.2f seconds.' % (end - start))
        
        if prune_rules:
            if verbose:
                print('Determining the average activation of each fuzzy logic rule on the training data...')
            start = time.time()
            o1 = self.input_layer(train_X)
            o2 = self.condition_layer(o1)
            o3 = self.rule_base_layer(o2, inference='product')
            mean_rule_activations = o3.mean(axis=0)
            end = time.time()
            if verbose:
                print('done; calculated in %.2f seconds.' % (end - start))
            plt.bar([x for x in range(len(mean_rule_activations))], mean_rule_activations)
            plt.show()

            if verbose:
                print('Removing weakly activated fuzzy logic rules...')
            indices = np.where(mean_rule_activations > np.mean(mean_rule_activations))[0]
            self.rules = [self.rules[index] for index in indices]
            self.weights = [self.weights[index] for index in indices]
            self.import_existing(self.rules, self.weights, self.antecedents, self.consequents)
            self.orphaned_term_removal()
            self.preprocessing()
            self.update()
            if verbose:
                print('done; removal of fuzzy logic rules has been reflected in the Neuro-Fuzzy network.')
        
        # prepare the Q-table
        self.network = TabularNetwork(self.get_number_of_rules(),
                                 self.action_set_length)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        if verbose:
            print('Transforming the trajectories to a compatible format...')
        start = time.time()
        rule_weights = []
        transformed_trajectories = []
        for trajectory in trajectories:
            state = trajectory[0]
            action_index = trajectory[1]
            reward = trajectory[2]
            next_state = trajectory[3]
            done = trajectory[4]
            
            # this code will only use the rule with the highest degree of activation to do updates
            self.truth_value(state)
            rule_index = np.argmax(self.current_rule_activations)
            rule_index_weight = self.current_rule_activations[rule_index]
            rule_weights.append(rule_index_weight)
            self.truth_value(next_state)
            next_rule_index = np.argmax(self.current_rule_activations)
            next_rule_index_weight = self.current_rule_activations[next_rule_index]
            transformed_trajectories.append((rule_index, action_index, reward, next_rule_index, done))
        end = time.time()
        if verbose:
            print('done; trajectories transformed in %.2f seconds.' % (end - start))
        
        if verbose:
            print('Begin Offline [Tabular] Conservative Q-Learning...')
        start = time.time()
        self.conservative_q_iteration(num_itrs=100, project_steps=50, cql_alpha=self.cql_alpha, sampled=True,
                                 training_dataset=transformed_trajectories, rule_weights=rule_weights)
        end = time.time()
        if verbose:
            print('done; completed in %.2f seconds.' % (end - start))
        
        # APFRB sensitivity analysis
        if apfrb_sensitivity_analysis:
            if verbose:
                print('Performing sensitivity analysis inspired by APFRB paper...')
            start = time.time()
            l_ks = []
            for k, _ in enumerate(self.rules):
                print(k)
                max_c_k = np.max(self.c_k(train_X, k))
                l_ks.append(max_c_k)
            end = time.time()
            if verbose:
                print('done; completed in %.2f seconds.' % (end - start))
            
            plt.bar([x for x in range(len(l_ks))], l_ks)
            plt.show()
            
            m_k_c_k = np.median(self.f3, axis=0) * l_ks
            
            plt.bar([x for x in range(len(m_k_c_k))], m_k_c_k)
            plt.show()
        
        # disable learning and exploration
        
        self.gamma=0.0
        self.alpha=0.0
        self.ee_rate=0.