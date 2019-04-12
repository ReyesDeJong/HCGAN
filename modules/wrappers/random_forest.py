import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
from sklearn.ensemble import RandomForestClassifier
from parameters import param_keys
import numpy as np

"""
RF wrapper, first designed for projections
"""


class RandomForest(object):

  def __init__(self, params_to_update=None):
    self.params = self.set_default_params()
    if params_to_update is not None:
      self.params.update(params_to_update)
    self.clf = self.get_clf()

  def get_clf(self):
    return RandomForestClassifier(**self.params[param_keys.CLF_PARAMS])

  def get_clf_default_params(self):
    clf_default_params = {'n_jobs': -1, 'n_estimators': 100,
                          'criterion': 'entropy',
                          'max_depth': 10, 'min_samples_leaf': 3,
                          'min_samples_split': 2,
                          'max_features': None}
    return clf_default_params

  def set_default_params(self):
    params = {
      param_keys.CLF_PARAMS: self.get_clf_default_params(),
      param_keys.N_IMPORTANT_FEATURE_TO_KEEP: 100
    }
    return params

  def fit(self, train_data_array, train_labels):
    self.clf.fit(X=train_data_array, y=train_labels)
    train_data_array = self.clf.transform(train_data_array)
    return train_data_array, train_labels

  def transform(self, data_array):
    most_relevant_features = self.get_n_most_important_features_from_array(
      data_array)
    return most_relevant_features

  def get_n_most_important_features_from_array(self, data_array):
    features_importance = np.array(self.clf.feature_importances_) \
                          / np.sum(self.clf.feature_importances_)
    most_important_indexs_sorted = np.argsort(features_importance)
    sorted_features = data_array[:, most_important_indexs_sorted]
    return sorted_features[:,
           :self.params[param_keys.N_IMPORTANT_FEATURE_TO_KEEP]]
