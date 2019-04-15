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
from feature_extraction.clf_over_FATS_tsfresh import \
  load_and_concatenate_features
import feature_extraction.FATS_extractor as FATS_extractor

REAL_DATA_NAME = 'catalina_north9classes'
SYN_DATA_NAME = 'catalina_amp_irregular_9classes_generated_10000'
SYN_DATA_FOLDER = os.path.join(PATH_TO_PROJECT, 'TSTR_data', 'generated',
                               'catalina_amp_irregular_9classes')
REAL_DATA_FOLDER = os.path.join(PATH_TO_PROJECT, 'TSTR_data',
                                'datasets_original', 'REAL', '9classes_100_100')
NAME_REAL_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'
NAME_REAL_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh.pkl'
NAME_SYN_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh_concatenated.pkl'
NAME_SYN_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'

if __name__ == '__main__':
  path_to_real_data = os.path.join(REAL_DATA_FOLDER, '%s.pkl' % REAL_DATA_NAME)
  path_to_syn_data = os.path.join(SYN_DATA_FOLDER,
                                  '%s.pkl' % SYN_DATA_NAME)
  real_data_loader = DataLoader(
      magnitude_key=general_keys.ORIGINAL_MAGNITUDE_RANDOM,
      time_key=general_keys.TIME_RANDOM,
      data_path=path_to_real_data)
  syn_data_loader = DataLoader(
      magnitude_key=general_keys.GENERATED_MAGNITUDE,
      time_key=general_keys.TIME,
      data_path=path_to_syn_data)
  x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
    real_data_loader.get_all_sets_data()#n_samples_to_get=10000)
  x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
    syn_data_loader.get_all_sets_data()#n_samples_to_get=10000)

  x_train_real2, y_train_real2, x_val_real2, y_val_real2, x_test_real2, y_test_real2 = \
    FATS_extractor.read_data_irregular_sampling(
        path_to_real_data, magnitude_key='original_magnitude_random',
        time_key='time_random')
  x_train_syn2, y_train_syn2, x_val_syn2, y_val_syn2, x_test_syn2, y_test_syn2 = \
    FATS_extractor.read_data_irregular_sampling(
        path_to_syn_data, magnitude_key='generated_magnitude',
        time_key='time')
  print(np.mean(x_train_real==x_train_real2))
  print(np.mean(y_train_real == y_train_real2))
  print(np.mean(x_train_syn==x_train_syn2))
  print(np.mean(y_train_syn == y_train_syn2))
  print(np.mean(y_test_real == y_test_real2))

  # #load features
  # path_to_real_fats_features = os.path.join(PATH_TO_PROJECT, REAL_DATA_FOLDER,
  #                                           NAME_REAL_FATS_FEATURES)
  # path_to_real_tsfresh_features = os.path.join(PATH_TO_PROJECT,
  #                                              REAL_DATA_FOLDER,
  #                                              NAME_REAL_TSFRESH_FEATURES)
  # path_to_syn_fats_features = os.path.join(PATH_TO_PROJECT, SYN_DATA_FOLDER,
  #                                          NAME_SYN_FATS_FEATURES)
  # path_to_syn_tsfresh_features = os.path.join(PATH_TO_PROJECT, SYN_DATA_FOLDER,
  #                                             NAME_SYN_TSFRESH_FEATURES)
  # real_merged_features = load_and_concatenate_features(
  #     [path_to_real_fats_features, path_to_real_tsfresh_features])
  # syn_merged_features = load_and_concatenate_features(
  #     [path_to_syn_fats_features, path_to_syn_tsfresh_features])
  #
  #
  # list_of_methods = [StandardScaler(), PCA(), TSNE()]
  # pipeline = Pipeline(list_of_methods)
  # projector = Projector(pipeline, show_plots=True)
  # projector.fit(x_train_real, y_train_real)
  # pipeline.print_dimensions_before_projection()
  # projector.project_and_plot_real_syn(
  #     x_train_real, y_train_real, x_train_syn, y_train_syn,
  #     save_fig_name='raw_curves_catalina')
