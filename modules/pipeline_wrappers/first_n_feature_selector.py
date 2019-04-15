import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
from parameters import param_keys
import numpy as np

"""
RF wrapper, first designed for projections
"""


class FirstNFeatSelector(object):

  def __init__(self, params_to_update=None):
    self.params = self.set_default_params()
    if params_to_update is not None:
      self.params.update(params_to_update)

  def set_default_params(self):
    params = {
      param_keys.N_FIRST_FEATURE_TO_KEEP: 1e10
    }
    return params

  def fit(self, train_data_array, train_labels):
    train_data_array = self.transform(train_data_array)
    return train_data_array, train_labels

  def transform(self, data_array):
    most_relevant_features = self.get_n_first_features_from_array(
      data_array)
    return most_relevant_features

  def get_n_first_features_from_array(self, data_array):
    return data_array[:, :self.params[param_keys.N_FIRST_FEATURE_TO_KEEP]]
