#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:41:25 2021

@author: john
"""

import numpy as np

from scipy.special import softmax
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import minkowski

def general_euclidean_distance(x, y):
    if len(x) == len(y):
        q = len(x)
        return minkowski(x, y, p=2) / np.power(q, 0.5)
    else:
        raise TypeError('The vectors must of of equal dimensionality in order to use the General Euclidean Distance metric.')

def RMSE(predicted_Y, target_Y):
    return np.sqrt(mean_squared_error(predicted_Y, target_Y))

def KL(predicted_Y, target_Y, tau=0.1):
    return (softmax(target_Y / tau, axis=1) * (np.log(softmax(target_Y / tau, axis=1) / softmax(predicted_Y, axis=1)))).sum()

def weighted_RMSE(predicted_Y, target_Y):
    # return KL(predicted_Y, target_Y)
    weights = np.max(target_Y, axis=1) - np.min(target_Y, axis=1)
    if np.all(weights == 0):
        return None
    else:
        est_actions = np.argmax(predicted_Y, axis=1)
        target_actions = np.argmax(target_Y, axis=1)
        comparisons = est_actions == target_actions
        # encoding = np.where(comparisons, -1, 1) # where -1 is good (we are trying to minimize)
        encoding = np.where(comparisons, 0, 1) # where 0 is good (we are trying to minimize)
        return np.multiply(encoding, weights).sum()
        # return np.sqrt(np.multiply(np.power((predicted_Y - target_Y), 2), np.reshape(weights, (weights.shape[0], 1))).sum() / predicted_Y.shape[0])

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

def boolean_indexing(v, fillval=np.nan):
    """
    Converts uneven list of lists to Numpy array with np.nan as padding for smaller lists.

    https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    fillval : TYPE, optional
        DESCRIPTION. The default is np.nan.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out
