import sys
import os

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import modules.utils as utils
import sklearn

"""
Pipeline method to transformate input data, first designed for projections
"""


class Pipeline(object):

  def __init__(self, list_of_pipeline_objects):
    self.list_of_pipeline_objects = list_of_pipeline_objects
    self.pipiline_dimensions = []

  """
  Run first to train all objects in pipeline
  """
  def fit(self, train_data_array, train_labels):
    self.pipiline_dimensions.append(train_data_array.shape)
    for object in self.list_of_pipeline_objects:
      train_data_array, train_labels = object.fit(
          train_data_array, train_labels)
      self.pipiline_dimensions.append(train_data_array.shape)
    return train_data_array, train_labels

  """
  After training objects of pipeline, run to get pipeline output for a test
  or val set
  """
  def transform(self, data_array):
    for object in self.list_of_pipeline_objects:
      data_array = object.transform(data_array)
    print('Dimensions before projection %s' % str(self._get_dimensions_before_projection()  ))
    return data_array

  def _get_dimensions_before_projection(self):
    return [None]+self.pipiline_dimensions[-1].shape[1:]
