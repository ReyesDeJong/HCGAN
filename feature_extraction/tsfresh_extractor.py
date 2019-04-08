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
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from modules.data_set_generic import Dataset

REAL_DATA_FOLDER = os.path.join('datasets_original', 'REAL', '9classes_100_100')
REAL_DATA_NAME = 'catalina_north9classes.pkl'  # 'starlight_new_bal_0.10.pkl'#
SAVE_NAME = 'catalina_north9classes_features_tsfresh.pkl'
VERBOSE = False
BATCH_SIZE = 20

"""
"""


def get_data_as_df(data):
    magnitudes, times = FATS_extractor.split_data_into_mag_and_time(data)
    ids = [np.full(magnitudes.shape[1], id + 1) for id in np.arange(magnitudes.shape[0])]
    time_for_ts_fresh = [np.arange(data.shape[1]) for id in np.arange(data.shape[0])]
    # labels_replicated = [np.full(x_train_real_mag.shape[1], label_val) for label_val in y_train_real]
    time_flatten = np.reshape(time_for_ts_fresh, (-1))  # time_for_ts_fresh, (-1))
    ids_flatten = np.reshape(ids, (-1))
    mag_flatten = np.reshape(magnitudes, (-1))
    time_stamp_flatten = np.reshape(times, (-1))
    dataset_dict = {
        'time': time_flatten,  # [idexes_to_get],
        'ids': ids_flatten,  # [idexes_to_get],
        'magnitude': mag_flatten,  # [idexes_to_get],
        'timestamp': time_stamp_flatten  # [idexes_to_get]
    }
    dataset_df = pd.DataFrame(dataset_dict, columns=list(dataset_dict.keys()))
    return dataset_df


def get_tsfresh(data):
    dataset = Dataset(data_array=data, data_labels=data, BATCH_SIZE=BATCH_SIZE)
    extraction_settings = ComprehensiveFCParameters()
    features_to_return = []
    start_time = time.time()
    eval_not_finished = 1
    while eval_not_finished != 0:
        # time_checked = check_times(times[i])
        data_batch = dataset.get_batch_eval()
        batch_df = get_data_as_df(data_batch)
        X = extract_features(batch_df,
                             column_id='ids', column_sort='time',
                             default_fc_parameters=extraction_settings,
                             impute_function=impute, n_jobs=-1)
        impute(X)
        fetures_batch = X.values
        features_to_return.append(fetures_batch)
        eval_not_finished = dataset.BATCH_COUNTER_EVAL
        if dataset.BATCH_COUNTER_EVAL % 100 == 0:
            time_usage = str(datetime.timedelta(
                seconds=int(round(time.time() - start_time))))
            print("it %i Time usage: %s" % (dataset.BATCH_COUNTER_EVAL, str(time_usage)), flush=True)
    features_to_return = np.array(features_to_return)
    time_usage = str(datetime.timedelta(
        seconds=int(round(time.time() - start_time))))
    print("Total Time usage: %s\n" % (str(time_usage)), flush=True)
    return features_to_return


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER, REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_data, magnitude_key='original_magnitude_random', time_key='time_random')

    train_features = get_tsfresh(x_train_real)
    val_features = get_tsfresh(x_val_real)
    test_features = get_tsfresh(x_test_real)

    pkl.dump({'train': train_features, 'val': val_features, 'test': test_features}, open(
        os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER,
                     SAVE_NAME), "wb"))
