import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', ))
sys.path.append(PATH_TO_PROJECT)
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
      save_fig_name=None, label_names=None):
    projected_data = self.pipeline.transform(data_array)
    self.plot_data_projection(projected_data, labels, title, save_fig_name,
                              label_names)

  def plot_data_projection(self, data_projection, labels, title, save_fig_name,
      label_names):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    self._silent_plot_data_projection(ax, data_projection, labels, title,
                                      label_names)
    self._save_fig(fig, save_fig_name)
    self._plt_show_wrapper()

  def _save_fig(self, fig, save_fig_name):
    if save_fig_name is not None:
      save_path = os.path.join(PATH_TO_PROJECT, 'results', 'projections')
      utils.check_dir(save_path)
      fig.savefig(os.path.join(save_path, '%s.png' % save_fig_name),
                  bbox_inches='tight')

  def _split_data_by_label(self, data_array, labels) -> dict:
    label_values = np.unique(labels).astype(int)
    data_by_label_dict = {}
    for single_label_value in label_values:
      single_label_idxs = np.where(labels == single_label_value)[0]
      data_by_label_dict[single_label_value] = data_array[single_label_idxs]
    return data_by_label_dict

  def project_and_plot_real_syn(self, real_data, real_labels, syn_data,
      syn_labels, title='Projection', save_fig_name=None, label_names=None):
    data_dict = self._real_syn_data_to_dict(real_data, real_labels, syn_data,
                                            syn_labels)
    data_dict = self._project_real_syn(data_dict)
    self._plot_real_syn_and_both(data_dict, title, save_fig_name, label_names)

  # TODO: how to avoid this kind of data manipulations [CODE SMELL]
  def _plot_real_syn_and_both(self, data_dict: dict, title, save_fig_name,
      label_names):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    self._silent_plot_data_projection(
        ax[0], data_dict[general_keys.SYN][general_keys.PROJECTED_DATA],
        data_dict[general_keys.SYN][general_keys.LABELS], 'GAN data',
        label_names)
    self._silent_plot_data_projection(
        ax[1], data_dict[general_keys.REAL][general_keys.PROJECTED_DATA],
        data_dict[general_keys.REAL][general_keys.LABELS], 'Real data',
        label_names)
    concatenated_projections = np.concatenate(
        [data_dict[general_keys.REAL][general_keys.PROJECTED_DATA],
         data_dict[general_keys.SYN][general_keys.PROJECTED_DATA]])
    concatenated_labels = np.concatenate(
        [data_dict[general_keys.REAL][general_keys.LABELS],
         data_dict[general_keys.SYN][general_keys.LABELS]])
    self._silent_plot_data_projection(
        ax[2], concatenated_projections,
        concatenated_labels, 'GAN + real data',
        label_names)
    self._set_plots_to_same_limits(ax)
    fig.suptitle(title, fontsize=14)
    self._save_fig(fig, save_fig_name)
    self._plt_show_wrapper()

  def _silent_plot_data_projection(self, ax, data_projection, labels, title,
      label_names):
    data_by_label_dict = self._split_data_by_label(data_projection, labels)
    for label_value in data_by_label_dict.keys():
      x = data_by_label_dict[label_value][:, 0]
      y = data_by_label_dict[label_value][:, 1]
      label_name = self._get_label_name(label_value, label_names)
      ax.scatter(x, y, alpha=0.5, edgecolors='none', label=label_name)
    ax.set_title(title)
    leg = ax.legend()
    for lh in leg.legendHandles:
      lh.set_alpha(1)
    return ax

  def _get_label_name(self, label_value, label_names):
    if label_names is None:
      return label_value
    else:
      return label_names[label_value]

  def _set_plots_to_same_limits(self, ax_plots_list):
    ax_x_lims = []
    ax_y_lims = []
    for ax in ax_plots_list:
      ax_x_lims += list(ax.get_xlim())
      ax_y_lims += list(ax.get_ylim())
    for ax in ax_plots_list:
      ax.set_xlim([np.min(ax_x_lims), np.max(ax_x_lims)])
      ax.set_ylim([np.min(ax_y_lims), np.max(ax_y_lims)])

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
