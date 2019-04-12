import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
from sklearn.decomposition import PCA

"""
PCA wrapper, first designed for projections
"""


class PCA(object):

  def __init__(self):
    self.pca = PCA()

  def fit(self, train_data_array, train_labels):
    self.pca.fit(train_data_array)
    train_data_array = self.pca.transform(train_data_array)
    return train_data_array, train_labels

  def transform(self, data_array):
    return self.pca.transform(data_array)
