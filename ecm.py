#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:23:46 2021

@author: john
"""

import sympy as sym
import matplotlib.pyplot as plt

from common import general_euclidean_distance
from sklearn.datasets import make_blobs
from sympy.solvers.solveset import linsolve

SUPPRESS_EXCEPTIONS = True

class Cluster:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.support = 1
    def add_support(self):
        self.support += 1

def ECM(X, Cs, Dthr, SUPPRESS_EXCEPTIONS=True):
    for i, x in enumerate(X):
        if len(Cs) == 0:
            """
            Step 0: Create the first cluster by simply taking the
            position of the first example from the input stream as the
            first cluster center Cc_{1}^{0}, and setting a value 0 for its cluster
            radius Ru_{1} [Fig. 2(a)].
            """
            C = Cluster(center=x, radius=0)
            Cs.append(C)
            
        """
        Step 1: If all examples of the data stream have been processed, the algorithm 
        is finished. Else, the current input example, $x_i$, is taken and the distances 
        between this example and all $n$ already created cluster 
        centers Cc_j, D_{ij} = ||x_{i} - Cc_{j}||, j = 1, 2, ..., n, are calculated.  
        """
        
        D_i = {} # distances between the i'th 'x' and the centers for each j'th cluster; dictionary is indexed by j
        for j, C in enumerate(Cs):
            D_i[j] = general_euclidean_distance(x, C.center)
            
            """
            Step 2: If there is any distance value, D_{ij} = ||x_{i} - Cc_{j}||, equal to, 
            or less than, at least one of the radii, Ru_{j}, j = 1, 2, ..., n, it means 
            that the current example belongs to a cluster C_{m} with the minimum distance
            
            D_{im} = ||x_{i} - Cc_{m}|| = min(||x_{i} - Cc_{j}||)
            
            subject to the constraint D_{ij} < Ru_{j},   j = 1, 2, ..., n.
            
            In this case, neither a new cluster is created, nor are any existing clusters 
            updated (the cases of $x_4$ and $x_6$ in Fig. 2); the algorithm returns to Step 1. 
            Else—go to the next step.
            """
            if D_i[j] < C.radius:
                C.add_support()
                break # the observation belongs to this cluster, C. Return to Step 1.
                
        if D_i[j] < C.radius:
            continue # the observation belongs to this cluster, C. Return to Step 1.
        
        """ 
        Step 3: Find cluster (with center Cc_{a} and cluster radius Ru_{a}) from all existing cluster 
        centers through calculating the values S_{ij} = D_{ij} + Ru_{j}, j = 1, 2, ..., n, and then 
        choosing the cluster center with the minimum value S_{ia}:
            
            S_{ia} = D_{ia} + Ru_{a} = min(S_{ij}), j = 1, 2, ..., n.
        """
        
        S_i = {}
        for item in D_i.items():
            j = item[0]
            D_j = item[1]
            C_j = Cs[j]
            Ru_j = C_j.radius
            S_i[j] = D_j + Ru_j
        a = min(S_i, key=S_i.get)
        S_ia = S_i[a]
        
        """
        Step 4: If S_{ia} is greater than $2 * Dthr$, the example $x_i$ does not belong to any 
        existing clusters. A new cluster is created in the same way as described in Step 0 
        (the cases of $x_3$ and $x_8$ in Fig. 2), and the algorithm returns to Step 1.
        """
        
        if S_ia > (2.0 * Dthr):
            """
            Step 0: Create the first cluster by simply taking the
            position of the first example from the input stream as the
            first cluster center Cc_{1}^{0}, and setting a value 0 for its cluster
            radius Ru_{1} [Fig. 2(a)].
            """
            C = Cluster(center=x, radius=0)
            Cs.append(C)
            continue # terminate further execution of this iteration
        else:
            """
            Step 5: If S_{ia} is not greater than $2 * Dthr$, the cluster $C_{a}$ is updated by moving 
            its center, Cc_{a}, and increasing the value of its radius, Ru_{a}. The updated 
            radius Ru_{a}^{new} is set to be equal to S_{ia} / 2 and the new center Cc_{a}^{new} is located
            at the point on the line connecting the $x_i$ and Cc_{a}, and the distance from the new 
            center Cc_{a}^{new} to the point $x_{i}$ is equal to Ru_{a}^{new} 
            (the cases of $x_2$, $x_5$, $x_7$ and $x_9$ in Fig. 2).
            The algorithm returns to Step 1
            """
            Ca = Cs[a]
            Ca.radius = S_ia / 2.0
            Ca.add_support()
            
            # parallel_vector = x - Ca.center
            # str_expression = [('%.2f + (%.2f * t)' % (x_i, v_i)) for x_i, v_i in zip(Ca.center, parallel_vector)]
            # # solve for t
            # t = sym.Symbol('t')
            # expression = sym.sympify(str_expression)
            # solution = linsolve(expression, t)
            # print(solution)
            # dist = general_euclidean_distance(Ca.center, x)
            # print(dist)
            # dist = general_euclidean_distance(Ca.center - 0.1 * parallel_vector, x)
            # print(dist)
            # print('current = %s' % D_i[a])
            # print('target = %s' % Ca.radius)
            
            n = Ca.support
            
            if n == 0:
                m_n_minus_1 = 0
            else:
                m_n_minus_1 = Ca.center
                
            # this is my custom implementation, it keeps a running mean approximation of the cluster center
            # the original paper uses the line between the cluster center, the observation, and the cluster radius
            # to update the cluster center. However, I found that in the above code (which is commented out), it sometimes
            # would be unable to find a solution. So, I instead chose the following route.
            
            # the error in this approximation can be found by running the following code:
            # dist = general_euclidean_distance(Ca.center, x)
            # print(Ca.radius - dist)
            
            Ca.center = ((n - 1) * Ca.center + x) / n
            
            if not SUPPRESS_EXCEPTIONS and general_euclidean_distance(Ca.center, x) != Ca.radius:
                raise Exception('The distance from the center of the relevant cluster is meant to be equal to the cluster\'s radius.')
    return Cs

def ECMc(X, Cs, Dthr):
    raise NotImplementedError('The offline ECMc function has not been implemented yet. It requires Mathwork\'s Constrained Minimization Method.')     