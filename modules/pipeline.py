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

  """
  Run first to train all objects in pipeline
  """
  def fit(self, train_data_array, train_labels):
    for object in self.list_of_pipeline_objects:
      train_data_array, train_labels = object.fit(
          train_data_array, train_labels)
    return train_data_array, train_labels

  """
  After training objects of pipeline, run to get pipeline output for a test
  or val set
  """
  def transform(self, data_array):
    for object in self.list_of_pipeline_objects:
      data_array = object.transform(data_array)
    return data_array
