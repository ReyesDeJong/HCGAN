import sys
import os
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import FATS
import numpy as np
import pickle as pkl
import sklearn
import time
import datetime
import feature_extraction.tinkering_FATS as FATS_extractor

REAL_DATA_FOLDER = os.path.join('generated', 'catalina_amp_irregular_9classes')
REAL_DATA_NAME = 'catalina_amp_irregular_9classes_generated_10000.pkl'#'starlight_new_bal_0.10.pkl'#
SAVE_NAME = 'catalina_north9classes_features.pkl'


"""
needed to change 
from scipy.interpolate import interp1d as interscipy 
in file FeatureFunctionLib.py
changed prod[k] to prod[int(k)] in FeatureFunctionLib.py->SlottedA_length->slotted_autocorrelation 

See some timestamp with repeated values
"""


def get_FATS(magnitudes, times, useful_features):
    filtered_a = FATS.FeatureSpace(featureList=useful_features,
                          Data=['magnitude', 'time'])
    fats_features = []
    start_time = time.time()
    for i in range(magnitudes.shape[0]):
        print(i)
        lc_aux = np.array([magnitudes[i], times[i]])
        features_obj = filtered_a.calculateFeature(lc_aux)
        features_results = np.array(features_obj.result('array'))
        fats_features.append(features_results)
        if i%100==0:
            time_usage = str(datetime.timedelta(
                seconds=int(round(time.time() - start_time))))
            print("it %i Time usage: %s" % (i, str(time_usage)), flush=True)
    fats_features = np.array(fats_features)
    time_usage = str(datetime.timedelta(
        seconds=int(round(time.time() - start_time))))
    print("Total Time usage: %s\n" % (str(time_usage)), flush=True)

    return fats_features


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER, REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_data,
            magnitude_key='generated_magnitude', time_key='time')
            #magnitude_key='original_magnitude_random', time_key='time_random')

    train_magnitudes, train_times = FATS_extractor.split_data_into_mag_and_time(x_train_real)
    val_magnitudes, val_times = FATS_extractor.split_data_into_mag_and_time(x_val_real)
    test_magnitudes, test_times = FATS_extractor.split_data_into_mag_and_time(x_test_real)

    useful_features_names = FATS_extractor.get_useful_FATS_features(train_magnitudes, train_times)

    train_features = FATS_extractor.get_FATS(train_magnitudes, train_times, useful_features_names)
    val_features = 0#get_FATS(val_magnitudes, val_times, useful_features_names)
    test_features = 0#get_FATS(test_magnitudes, test_times, useful_features_names)


    pkl.dump({'train': train_features, 'val': val_features, 'test': test_features}, open(
        os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER,
                     SAVE_NAME), "wb"))






