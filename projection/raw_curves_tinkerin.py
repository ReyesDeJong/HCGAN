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

REAL_DATA_NAME = 'starlight_new_bal_1.00.pkl'

if __name__ == '__main__':
  path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data',
                                   'datasets_original', 'REAL', REAL_DATA_NAME)
  data_loader = DataLoader(
      magnitude_key=general_keys.ORIGINAL_MAGNITUDE, time_key=general_keys.TIME,
      data_path=path_to_real_data)
  x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
    data_loader.get_all_sets_data()
  index_to_get_val = 100
  x_train_real = x_train_real[:index_to_get_val, :, 0]
  y_train_real = y_train_real[:index_to_get_val]

  list_of_methods = [StandardScaler(), PCA(), TSNE()]
  pipeline = Pipeline(list_of_methods)
  projector = Projector(pipeline)
  projector.fit(x_train_real, y_train_real)
  projector.project_and_plot_data(x_train_real, y_train_real,
                                  save_fig_name='try')
  projector.project_and_plot_real_syn(
      x_train_real, y_train_real, x_train_real, y_train_real,
      save_fig_name='nice')
