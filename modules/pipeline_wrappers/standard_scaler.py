import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
import sklearn.preprocessing as preprocessing

"""
Standar scaler wrapper, first designed for projections
"""


class StandardScaler(object):

  def __init__(self):
    self.scaler = preprocessing.StandardScaler()

  def fit(self, train_data_array, train_labels):
    self.scaler.fit(train_data_array)
    train_data_array = self.scaler.transform(train_data_array)
    return train_data_array, train_labels

  def transform(self, data_array):
    return self.scaler.transform(data_array)
