import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import matplotlib.pyplot as plt
from modules.wrappers.standard_scaler import StandardScaler
from modules.wrappers.pca import PCA
from modules.wrappers.tsne import TSNE
from modules.data_loaders.data_loader import DataLoader
import parameters.general_keys as general_keys
import parameters.param_keys as param_keys
from modules.pipeline import Pipeline
from projection.projector import Projector

DATA_NAME = 'starlight_new_bal_1.00'
SYN_DATA_FOLDER = os.path.join(PATH_TO_PROJECT, 'TSTR_data', 'generated',
                               DATA_NAME)
REAL_DATA_FOLDER = os.path.join(PATH_TO_PROJECT, 'TSTR_data',
                                'datasets_original', 'REAL')

if __name__ == '__main__':
  path_to_real_data = os.path.join(REAL_DATA_FOLDER, '%s.pkl' % DATA_NAME)
  path_to_syn_data = os.path.join(SYN_DATA_FOLDER,
                                  '%s_generated.pkl' % DATA_NAME)
  real_data_loader = DataLoader(
      magnitude_key=general_keys.ORIGINAL_MAGNITUDE, time_key=general_keys.TIME,
      data_path=path_to_real_data)
  syn_data_loader = DataLoader(
      magnitude_key=general_keys.ORIGINAL_MAGNITUDE, time_key=general_keys.TIME,
      data_path=path_to_real_data)
  x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
    real_data_loader.get_all_sets_data()
  x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
    syn_data_loader.get_all_sets_data()
  # get magnitudes only
  x_train_real = x_train_real[..., 0]
  x_train_syn = x_train_syn[..., 0]

  tsne_param = {param_keys.VERBOSE: 1}
  list_of_methods = [StandardScaler(), PCA(), TSNE(tsne_param)]
  pipeline = Pipeline(list_of_methods)
  projector = Projector(pipeline, show_plots=False)
  projector.fit(x_train_real, y_train_real)
  pipeline.print_dimensions_before_projection()
  projector.project_and_plot_real_syn(
      x_train_real, y_train_real, x_train_syn, y_train_syn,
      save_fig_name='nice')
