import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
import sklearn.manifold as manifold

"""
Standar scaler wrapper, first designed for projections
"""


class TSNE(object):

  def __init__(self, params_to_update=None):
    self.params = self.set_default_params()
    if params_to_update is not None:
      self.params.update(params_to_update)
    self.tsne = manifold.TSNE(**self.params)

  def set_default_params(self):
    params = {
      'n_components': 2,
      'verbose': 0,
      'perplexity': 40,
      'n_iter': 300
    }
    return params

  def fit(self, train_data_array, train_labels):
    return train_data_array, train_labels

  def transform(self, data_array):
    return self.tsne.fit_transform(data_array)
