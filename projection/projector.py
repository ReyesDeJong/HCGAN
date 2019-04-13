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
import parameters.general_keys as general_keys

"""
Data projector, given pipeline
"""


class Projector(object):
  def __init__(self, pipeline: Pipeline, show_plots=True):
    self.pipeline = pipeline
    self.show_plot = show_plots

  def _plt_show_wrapper(self):
    if self.show_plot:
      plt.show()
    else:
      plt.close()

  def fit(self, train_data_array, train_labels):
    self.pipeline.fit(train_data_array, train_labels)

  def project_data(self, data_array):
    return self.pipeline.transform(data_array)

  def project_and_plot_data(self, data_array, labels, title='Projection',
      save_fig_name=None):
    projected_data = self.pipeline.transform(data_array)
    self.plot_data_projection(projected_data, labels, title, save_fig_name)

  def plot_data_projection(self, data_projection, labels, title, save_fig_name):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    self._silent_plot_data_projection(ax, data_projection, labels, title)
    self._save_fig(fig, save_fig_name)
    self._plt_show_wrapper()

  def _save_fig(self, fig, save_fig_name):
    if save_fig_name is not None:
      save_path = os.path.join(PATH_TO_PROJECT, 'results', 'projections')
      utils.check_dir(save_path)
      fig.savefig(os.path.join(save_path, '%s.png' % save_fig_name),
                  bbox_inches='tight')

  def _split_data_by_label(self, data_array, labels) -> dict:
    label_values = np.unique(labels)
    data_by_label_dict = {}
    for single_label_value in label_values:
      single_label_idxs = np.where(labels == single_label_value)[0]
      data_by_label_dict[single_label_value] = data_array[single_label_idxs]
    return data_by_label_dict

  def project_and_plot_real_syn(self, real_data, real_labels, syn_data,
      syn_labels, title='Projection', save_fig_name=None):
    data_dict = self._real_syn_data_to_dict(real_data, real_labels, syn_data,
                                            syn_labels)
    data_dict = self._project_real_syn(data_dict)
    self._plot_real_syn_and_both(data_dict, title, save_fig_name)

  # TODO: how to avoid this kind of data manipulations [CODE SMELL]
  def _plot_real_syn_and_both(self, data_dict: dict, title, save_fig_name):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    self._silent_plot_data_projection(
        ax[0], data_dict[general_keys.SYN][general_keys.PROJECTED_DATA],
        data_dict[general_keys.SYN][general_keys.LABELS], 'GAN data')
    self._silent_plot_data_projection(
        ax[1], data_dict[general_keys.REAL][general_keys.PROJECTED_DATA],
        data_dict[general_keys.REAL][general_keys.LABELS], 'Real data')
    concatenated_projections = np.concatenate(
        [data_dict[general_keys.REAL][general_keys.PROJECTED_DATA],
         data_dict[general_keys.SYN][general_keys.PROJECTED_DATA]])
    concatenated_labels = np.concatenate(
        [data_dict[general_keys.REAL][general_keys.LABELS],
         data_dict[general_keys.SYN][general_keys.LABELS]])
    self._silent_plot_data_projection(
        ax[2], concatenated_projections,
        concatenated_labels, 'GAN + real data')
    fig.suptitle(title, fontsize=14)
    self._save_fig(fig, save_fig_name)
    self._plt_show_wrapper()

  def _silent_plot_data_projection(self, ax, data_projection, labels, title):
    data_by_label_dict = self._split_data_by_label(data_projection, labels)
    for label in data_by_label_dict.keys():
      x = data_by_label_dict[label][:, 0]
      y = data_by_label_dict[label][:, 1]
      ax.scatter(x, y, alpha=0.8, edgecolors='none', label=label)
    ax.set_title(title)
    ax.legend()
    return ax

  # TODO: how to avoid this kind of data manipulations [CODE SMELL]
  def _project_real_syn(self, data_dict: dict):
    data_to_project = np.concatenate(
        [data_dict[general_keys.REAL][general_keys.DATA_ARRAY],
         data_dict[general_keys.SYN][general_keys.DATA_ARRAY]])
    projected_data = self.pipeline.transform(data_to_project)
    data_dict[general_keys.REAL][general_keys.PROJECTED_DATA] = \
      projected_data[
      :data_dict[general_keys.REAL][general_keys.DATA_ARRAY].shape[0]]
    data_dict[general_keys.SYN][general_keys.PROJECTED_DATA] = \
      projected_data[
      data_dict[general_keys.REAL][general_keys.DATA_ARRAY].shape[0]:]
    assert data_dict[general_keys.SYN][general_keys.PROJECTED_DATA].shape[0] == \
           data_dict[general_keys.SYN][general_keys.DATA_ARRAY].shape[0]
    return data_dict

  def _real_syn_data_to_dict(self, real_data, real_labels, syn_data,
      syn_labels) -> dict:
    data_dict = {general_keys.REAL: {general_keys.DATA_ARRAY: real_data,
                                     general_keys.LABELS: real_labels},
                 general_keys.SYN: {general_keys.DATA_ARRAY: syn_data,
                                    general_keys.LABELS: syn_labels}}
    return data_dict
