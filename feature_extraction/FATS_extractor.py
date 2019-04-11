import sys
import os
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
from modules import FATS
import numpy as np
import pickle as pkl
import sklearn
import time
import datetime




REAL_DATA_FOLDER = os.path.join('datasets_original', 'REAL', '9classes_100_100')
REAL_DATA_NAME = 'catalina_north9classes.pkl'#'starlight_new_bal_0.10.pkl'#
SAVE_NAME = 'catalina_north9classes_features.pkl'
VERBOSE = False

"""
needed to change 
from scipy.interpolate import interp1d as interscipy 
in file FeatureFunctionLib.py
changed prod[k] to prod[int(k)] in FeatureFunctionLib.py->SlottedA_length->slotted_autocorrelation 

See some timestamp with repeated values

NASTY TIME FIXING
"""


def get_data_from_set(set, magnitude_key, time_key):
    magnitudes = np.asarray(set[magnitude_key])
    time = np.asarray(set[time_key])
    x = np.stack((magnitudes, time), axis=-1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = np.asarray(set['class'])
    x, y = sklearn.utils.shuffle(x, y, random_state=42)
    return x, y

def load_pickle(path):
    infile = open(path, 'rb')
    dataset_partitions = pkl.load(infile)
    return dataset_partitions

def read_data_irregular_sampling(file, magnitude_key='original_magnitude', time_key='time', verbose=False):
    dataset_partitions = np.load(file)
    # dataset_partitions = np.load(file)
    if verbose:
        print(dataset_partitions[0].keys())
    x_train, y_train = get_data_from_set(dataset_partitions[0], magnitude_key, time_key)
    x_val, y_val = get_data_from_set(dataset_partitions[1], magnitude_key, time_key)
    x_test, y_test = get_data_from_set(dataset_partitions[2], magnitude_key, time_key)
    return x_train, y_train, x_val, y_val, x_test, y_test

def split_data_into_mag_and_time(data_array):
    magnitude = data_array[..., 0]
    time = data_array[..., 1]
    return magnitude, time

def get_data_fraction_as_mag_and_time(data_array, n_samples_to_get=3):
    return split_data_into_mag_and_time(data_array[:n_samples_to_get])


#TODO: check why s this necessary
def repair_time(time):
    previous_time_step = time[0]
    for time_step_idx in range(time.shape[0] - 1):
        time_step_to_be_compered = time[time_step_idx + 1]
        if time_step_to_be_compered == previous_time_step:
            time[time_step_idx + 1] = previous_time_step + 1e-7
            #print('unique', np.unique(np.sort(time)).shape)
            #print(time_step_idx)
            #print(time_step_to_be_compered, '==', previous_time_step)
            #print(time[time_step_idx + 1])
            #print(time_step_to_be_compered + 2e-8)
            #print('sort', np.sort(time))
            return np.sort(time)
        previous_time_step = time[time_step_idx + 1]
    return time

def check_times(time, verbose=VERBOSE):
    original_time_shape = time.shape[0]
    if verbose:
        if np.unique(time).shape[0] != original_time_shape:
            print('NASTY TIME >:(')
    while np.unique(time).shape[0] != original_time_shape:
        time = repair_time(time)
    return time


def get_useful_FATS_features(magnitudes, times):
    #print('time orig', times[0])
    time_checked = check_times(times[0])
    #print('time check', time)
    lc_example = np.array([magnitudes[0], time_checked])
    a = FATS.FeatureSpace(Data=['magnitude', 'time'])
    b = a.calculateFeature(lc_example)
    feature_list = np.array(list(b.result('dict').keys()))
    feature_values = np.array(list(b.result('dict').values()))
    useful_features = list(feature_list[np.argwhere(~np.isnan(feature_values.astype(np.float64)))])#list(feature_list[np.where(feature_values != None)])
    print('%i features calculated' % len(useful_features))
    print([x[0] for x in useful_features])
    return useful_features

def get_FATS(magnitudes, times, useful_features):
    filtered_a = FATS.FeatureSpace(featureList=useful_features,
                                   Data=['magnitude', 'time'])
    fats_features = []
    start_time = time.time()
    for i in range(magnitudes.shape[0]):
        time_checked = check_times(times[i])
        lc_aux = np.array([magnitudes[i], time_checked])
        features_obj = filtered_a.calculateFeature(lc_aux)
        features_results = np.array(features_obj.result('array'))
        fats_features.append(features_results)
        if i%100==0:
            time_usage = str(datetime.timedelta(
                seconds=int(round(time.time() - start_time))))
            print("it %i Time usage: %s" % (i, str(time_usage)), flush=True)
    fats_features = np.array(fats_features) #shouled this be concatenate?
    time_usage = str(datetime.timedelta(
        seconds=int(round(time.time() - start_time))))
    print("Total Time usage: %s\n" % (str(time_usage)), flush=True)

    return fats_features


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER, REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        read_data_irregular_sampling(
            path_to_real_data, magnitude_key='original_magnitude_random', time_key='time_random')

    train_magnitudes, train_times = split_data_into_mag_and_time(x_train_real)
    val_magnitudes, val_times = split_data_into_mag_and_time(x_val_real)
    test_magnitudes, test_times = split_data_into_mag_and_time(x_test_real)

    useful_features_names = get_useful_FATS_features(train_magnitudes, train_times)

    train_features = get_FATS(train_magnitudes, train_times, useful_features_names)
    val_features = get_FATS(val_magnitudes, val_times, useful_features_names)
    test_features = get_FATS(test_magnitudes, test_times, useful_features_names)


    pkl.dump({'train': train_features, 'val': val_features, 'test': test_features}, open(
        os.path.join(PATH_TO_PROJECT, 'TSTR_data', REAL_DATA_FOLDER,
                     SAVE_NAME), "wb"))






