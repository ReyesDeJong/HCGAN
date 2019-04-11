import sys
import os

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import pickle as pkl
import pandas as pd
import sklearn
import time
import datetime
import feature_extraction.tinkering_FATS as FATS_extractor
import feature_extraction.tsfresh_extractor as tsfresh_extractor

REAL_DATA_FOLDER = os.path.join('generated', 'catalina_amp_irregular_9classes')
REAL_DATA_NAME = 'catalina_amp_irregular_9classes_generated_10000.pkl'#'starlight_new_bal_0.10.pkl'#
SAVE_NAME = 'catalina_north9classes_features_tsfresh_2.pkl'
BATCH_SIZE = 100


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER, REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_data, magnitude_key='generated_magnitude', time_key='time')

    train_features = tsfresh_extractor.get_tsfresh(x_train_real[45000:])
    val_features = 0#tsfresh_extractor.get_tsfresh(x_val_real)
    test_features = 0#tsfresh_extractor.get_tsfresh(x_test_real)

    pkl.dump({'train': train_features, 'val': val_features, 'test': test_features}, open(
        os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER,
                     SAVE_NAME), "wb"))
