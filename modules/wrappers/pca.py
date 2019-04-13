import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
import sklearn.decomposition as decomposition
from parameters import param_keys
import numpy as np

"""
PCA wrapper, first designed for projections
"""


class PCA(object):

  def __init__(self, params_to_update=None):
    self.params = self.set_default_params()
    if params_to_update is not None:
      self.params.update(params_to_update)
    self.pca = decomposition.PCA()

  def set_default_params(self):
    params = {
      param_keys.VARIANCE_PERCENTAGE_TO_KEEP: 0.95
    }
    return params

  def fit(self, train_data_array, train_labels):
    self.pca.fit(X=train_data_array, y=train_labels)
    train_data_array = self.transform(train_data_array)
    return train_data_array, train_labels

  def transform(self, data_array):
    data_array_pca = self.pca.transform(data_array)
    most_relevant_features = self.get_most_important_features_from_array(
        data_array_pca)
    return most_relevant_features

  def get_most_important_features_from_array(self, data_array):
    variance_percentage = self.pca.explained_variance_ratio_
    most_important_indexs_sorted = np.argsort(variance_percentage)[::-1]
    sorted_features = data_array[:, most_important_indexs_sorted]
    variance_percentage_sorted = variance_percentage[
      most_important_indexs_sorted]
    cum_sum_variance = np.cumsum(variance_percentage_sorted)
    indx_important_pca_components = np.argmax(
        cum_sum_variance > self.params[param_keys.VARIANCE_PERCENTAGE_TO_KEEP])
    return sorted_features[:, :indx_important_pca_components]
