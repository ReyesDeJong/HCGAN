import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', ))
sys.path.append(PATH_TO_PROJECT)
import sklearn.decomposition as decomposition
from parameters import param_keys
from modules.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import modules.utils as utils

"""
Data projector, given pipeline
"""


class Projector(object):
  def __init__(self, pipeline: Pipeline):
    self.pipeline = pipeline

  def fit(self, train_data_array, train_labels):
    self.pipeline.fit(X=train_data_array, y=train_labels)

  def project_data(self, data_array):
    return self.pipeline.transform(data_array)

  def project_and_plot_data(self, data_array, labels, title='Projection',
      save_fig_name=None):
    projected_data = self.pipeline.transform(data_array)
    self.plot_data_projection(projected_data, labels, title, save_fig_name)

  def plot_data_projection(self, data_projection, labels, title, save_fig_name):
    data_by_label_dict = self.split_data_by_label(data_projection, labels)
    fig = plt.figure()
    for label in data_by_label_dict.keys():
      x = data_by_label_dict[label][:, 0]
      y = data_by_label_dict[label][:, 1]
      plt.scatter(x, y, alpha=0.8, edgecolors='none', label=label)
    plt.title(title)
    plt.legend()
    self._save_fig(fig, save_fig_name)
    plt.show()
    plt.close()

  def _save_fig(self, fig, save_fig_name):
    if save_fig_name is not None:
      save_path = os.path.join(PATH_TO_PROJECT, 'results', 'projections')
      utils.check_dir(save_path)
      fig.savefig(os.path.join(save_path, '%s.png' % save_fig_name))

  def split_data_by_label(self, data_array, labels) -> dict:
    label_values = np.unique(labels)
    data_by_label_dict = {}
    for single_label_value in label_values:
      single_label_idxs = np.where(labels == single_label_value)[0]
      data_by_label_dict[single_label_value].append(
          data_array[single_label_idxs])
    data_by_label_dict
