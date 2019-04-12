import sys
import os
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import modules.utils as utils
import sklearn

"""
Base class to oad different types of data, in this case for Starlight and
 Catalina Light curves
"""


class DataLoader(object):

  def __init__(self, magnitude_key, time_key, data_path):
    self.magnitude_key = magnitude_key
    self.time_key = time_key
    self.data_path = data_path
    self.verbose = False

  def load_data(self, path):
    return utils.load_pickle(path)

  def set_verbose(self, verbose):
    self.verbose = verbose

  def get_data_from_set(self, set_partition):
    magnitudes = np.asarray(set_partition[self.magnitude_key])
    magnitudes = np.squeeze(magnitudes)
    time = np.asarray(set_partition[self.time_key])
    time = np.squeeze(time)
    y = np.asarray(set_partition['class'])
    magnitudes, time, y = sklearn.utils.shuffle(magnitudes, time, y,
                                                random_state=42)
    return magnitudes, time, y

  """
  magnitude and time are stacked as a channel in the end
  """
  def get_data_from_set_as_x_y(self, set):
    magnitudes, time, y = self.get_data_from_set(set)
    x = np.stack((magnitudes, time), axis=-1)
    return x, y


  def get_all_sets_data(self):
    dataset_partitions = self.load_data(self.data_path)
    if self.verbose: #TODO: verbose as printr manager not as a condition
        print(dataset_partitions[0].keys())
    x_train, y_train = self.get_data_from_set_as_x_y(
        dataset_partitions[0], self.magnitude_key, self.time_key)
    x_val, y_val = self.get_data_from_set_as_x_y(
        dataset_partitions[1], self.magnitude_key, self.time_key)
    x_test, y_test = self.get_data_from_set_as_x_y(
        dataset_partitions[2], self.magnitude_key, self.time_key)
    return x_train, y_train, x_val, y_val, x_test, y_test
