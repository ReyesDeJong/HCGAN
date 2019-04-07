import pandas as pd
import sys
import os

# sys.path.append("/home/rodrigo/FATS-2.0/")
sys.path.append("../../FATS-2.0/")
import FATS
import numpy as np
import time
import itertools
import pickle as pkl
import h5py
import sklearn

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

REAL_DATA_NAME = 'catalina_north9classes.pkl'#'starlight_new_bal_0.10.pkl'#

"""
needed to change 
from scipy.interpolate import interp1d as interscipy 
in file FeatureFunctionLib.py
changed prod[k] to prod[int(k)] in FeatureFunctionLib.py->SlottedA_length->slotted_autocorrelation 

See some timestamp with repeated values
"""


def get_data_from_set(set, magnitude_key, time_key):
    magnitudes = np.asarray(set[magnitude_key])
    time = np.asarray(set[time_key])
    x = np.stack((magnitudes, time), axis=-1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = np.asarray(set['class'])
    x, y = sklearn.utils.shuffle(x, y, random_state=42)
    return x, y


def read_data_irregular_sampling(file, magnitude_key='original_magnitude', time_key='time', verbose=False):
    infile = open(file, 'rb')
    dataset_partitions = pkl.load(infile)
    # dataset_partitions = np.load(file)
    if verbose:
        print(dataset_partitions[0].keys())
    x_train, y_train = get_data_from_set(dataset_partitions[0], magnitude_key, time_key)
    x_val, y_val = get_data_from_set(dataset_partitions[1], magnitude_key, time_key)
    x_test, y_test = get_data_from_set(dataset_partitions[2], magnitude_key, time_key)
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_data_fraction_as_mag_and_time(data_array, n_samples_to_get=3):
    magnitude = data_array[:n_samples_to_get, :, 0]
    time = data_array[:n_samples_to_get, :, 1]
    return magnitude, time


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', 'datasets_original', 'REAL', REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        read_data_irregular_sampling(
            path_to_real_data, magnitude_key='original_magnitude_random', time_key='time_random')

    train_mag, train_time = get_data_fraction_as_mag_and_time(x_train_real)

    lc_example = np.array([train_mag[0], train_time[0]])

    a = FATS.FeatureSpace(Data=['magnitude', 'time'])
    b = a.calculateFeature(lc_example)
    results = np.array(b.result('array'))

    feature_list = np.array(list(b.result('dict').keys()))
    feature_values = np.array(list(b.result('dict').values()))
    useful_features = list(feature_list[np.argwhere(~np.isnan(feature_values.astype(np.float64)))])#list(feature_list[np.where(feature_values != None)])

    filtered_a = FATS.FeatureSpace(featureList=useful_features,
                          Data=['magnitude', 'time'])
    filtered_b = filtered_a.calculateFeature(lc_example)
    results_filtered = np.array(filtered_b.result('array'))




