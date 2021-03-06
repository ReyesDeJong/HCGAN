import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import matplotlib.pyplot as plt
from modules.pipeline_wrappers.standard_scaler import StandardScaler
from modules.pipeline_wrappers.pca import PCA
from modules.pipeline_wrappers.tsne import TSNE
from modules.pipeline_wrappers.lightGBM import LightGBM
from modules.data_loaders.data_loader import DataLoader
from modules.pipeline_wrappers.first_n_feature_selector import \
  FirstNFeatSelector
import parameters.general_keys as general_keys
import parameters.param_keys as param_keys
from modules.pipeline import Pipeline
from projection.projector import Projector
from feature_extraction.clf_over_FATS_tsfresh import \
  load_and_concatenate_features
import feature_extraction.FATS_extractor as FATS_extractor

"""
IMPORTANT NOTE: loading cannot be performed without shuffle ramdom=42, because
features were calculated in that order, and any othar one woul cause a missmatch
Its recomended to have a single pickle with everything.
"""

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
N_SAMPLES_TO_TRAIN_PROJECTION = int(1e100)
N_SAMPLES_TO_PROJECT = int(10000)

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
    real_data_loader.get_all_sets_data(
        n_samples_to_get=N_SAMPLES_TO_TRAIN_PROJECTION)
  x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
    syn_data_loader.get_all_sets_data(
        n_samples_to_get=N_SAMPLES_TO_TRAIN_PROJECTION)

  # load features
  path_to_real_fats_features = os.path.join(PATH_TO_PROJECT, REAL_DATA_FOLDER,
                                            NAME_REAL_FATS_FEATURES)
  path_to_real_tsfresh_features = os.path.join(PATH_TO_PROJECT,
                                               REAL_DATA_FOLDER,
                                               NAME_REAL_TSFRESH_FEATURES)
  path_to_syn_fats_features = os.path.join(PATH_TO_PROJECT, SYN_DATA_FOLDER,
                                           NAME_SYN_FATS_FEATURES)
  path_to_syn_tsfresh_features = os.path.join(PATH_TO_PROJECT, SYN_DATA_FOLDER,
                                              NAME_SYN_TSFRESH_FEATURES)

  real_merged_features = load_and_concatenate_features(
      [path_to_real_fats_features, path_to_real_tsfresh_features])[
                           general_keys.TRAIN_SET][
                         :N_SAMPLES_TO_TRAIN_PROJECTION]
  syn_merged_features = load_and_concatenate_features(
      [path_to_syn_fats_features, path_to_syn_tsfresh_features])[
                          general_keys.TRAIN_SET][
                        :N_SAMPLES_TO_TRAIN_PROJECTION]

  clf_params = {param_keys.N_IMPORTANT_FEATURE_TO_KEEP: 100}
  pca_params = {param_keys.VARIANCE_PERCENTAGE_TO_KEEP: 0.80}
  selector_params = {param_keys.N_FIRST_FEATURE_TO_KEEP: 10}
  list_of_methods = [StandardScaler(), LightGBM(clf_params),
                     StandardScaler(),
                     PCA(),
                     #FirstNFeatSelector(selector_params),
                     TSNE()]
  pipeline = Pipeline(list_of_methods)
  projector = Projector(pipeline, show_plots=True)
  projector.fit(real_merged_features, y_train_real)
  pipeline.print_dimensions_before_projection()
  projector.project_and_plot_real_syn(
      real_merged_features[:N_SAMPLES_TO_PROJECT],
      y_train_real[:N_SAMPLES_TO_PROJECT],
      syn_merged_features[:N_SAMPLES_TO_PROJECT],
      y_train_syn[:N_SAMPLES_TO_PROJECT],
      save_fig_name='features_catalina_mo_selector_85var',
      title='Catalina with 10 most important features from LGBM+PCA',
      label_names=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd',
                   'beta Lyrae', 'HADS'])
