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

# get the current working directory, but only keep the parent folder (which is 'fuzzy')
path = os.getcwd() + '/fuzzy'
# ignore any directory that has '.' in it (e.g. .gitignore)
directories = [folder for folder in os.listdir(path) if '.' not in folder]

for directory in directories:
    sys.path.append(path + '/' + directory)

from cfql import CFQLModel
from constant import PROBLEM_LIST

class Model:
    def __init__(self):
        self.lookup_table = {}
    
    def build(self):
        self.lookup_table['problem'] = CFQLModel().load()