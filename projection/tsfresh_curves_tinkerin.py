import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from feature_extraction.tinkering_FATS import load_pickle

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

REAL_DATA_NAME = 'starlight_new_bal_0.10.pkl'

def targets_to_numbers(targets):
    target_keys = np.unique(targets)
    target_keys_idxs = np.argsort(np.unique(targets))
    targets_as_numbers = target_keys_idxs[np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]
    return targets_as_numbers

def get_data_from_set(set, magnitude_key, time_key):
    magnitudes = np.asarray(set[magnitude_key])
    time = np.asarray(set[time_key])
    x = np.stack((magnitudes, time), axis=-1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = np.asarray(set['class'])
    x, y = shuffle(x, y, random_state=42)
    y = targets_to_numbers(y)
    return x, y


def read_data_irregular_sampling(file, magnitude_key='original_magnitude', time_key='time', verbose=False):
    dataset_partitions = load_pickle(file)#np.load(file)
    if verbose:
        print(dataset_partitions[0].keys())
    x_train, y_train = get_data_from_set(dataset_partitions[0], magnitude_key, time_key)
    x_val, y_val = get_data_from_set(dataset_partitions[1], magnitude_key, time_key)
    x_test, y_test = get_data_from_set(dataset_partitions[2], magnitude_key, time_key)
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == '__main__':
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', 'datasets_original', 'REAL', REAL_DATA_NAME)
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        read_data_irregular_sampling(
        path_to_real_data, magnitude_key='original_magnitude', time_key='time')

    x_train_real = x_val_real[:100]
    y_train_real = y_val_real[:100]

    x_train_real_mag = x_train_real[:, :, 0]
    x_train_real_time = x_train_real[:, :, 1]
    ids = [np.full(x_train_real_mag.shape[1], id+1) for id in np.arange(x_train_real.shape[0])]
    time_for_ts_fresh = [np.arange(x_train_real.shape[1]) for id in np.arange(x_train_real.shape[0])]
    labels_replicated = [np.full(x_train_real_mag.shape[1], label_val) for label_val in y_train_real]

    time_flatten = np.reshape(time_for_ts_fresh, (-1))
    ids_flatten = np.reshape(ids, (-1))
    mag_flatten = np.reshape(x_train_real_mag, (-1))
    time_stamp_flatten = np.reshape(x_train_real_time, (-1))

    idexes_to_get = np.arange(len(time_for_ts_fresh))
    np.random.shuffle(idexes_to_get)

    #ids_flatten = [str(id) for id in ids_flatten]

    dataset_dict = {
            'time': time_flatten, #[idexes_to_get],
            'ids': ids_flatten, #[idexes_to_get],
            'magnitude': mag_flatten, #[idexes_to_get],
            'timestamp': time_stamp_flatten#[idexes_to_get]
            }

    dataset_df = pd.DataFrame(dataset_dict, columns=list(dataset_dict.keys()))

    extraction_settings = ComprehensiveFCParameters()

    X = extract_features(dataset_df,
                         column_id='ids', column_sort='time',
                         default_fc_parameters=extraction_settings,
                         impute_function=impute, n_jobs=-1)

    impute(X)
    y = pd.Series(y_train_real, index=np.arange(y_train_real.shape[0])+1)
    features_filtered = select_features(X, y)

    x_train_real = features_filtered.values
    scaler = StandardScaler()
    scaler.fit(x_train_real)
    x_train_scaled = scaler.transform(x_train_real)
    #
    # pca = PCA()
    # pca.fit(x_train_scaled)
    # x_train_pca = pca.transform(x_train_scaled)
    #
    # variance_precentage = pca.explained_variance_ratio_
    # cum_sum_variance = np.cumsum(variance_precentage)
    # indx_important_pca_components = np.argmax(cum_sum_variance > 0.9)
    #
    # indexes_of_array = np.arange(x_train_pca.shape[0])
    # np.random.shuffle(indexes_of_array)
    # index_to_get_val = 5000
    # index_to_get = indexes_of_array[:index_to_get_val]
    x_train_pca_to_plot = x_train_scaled#x_train_pca[:,:indx_important_pca_components][index_to_get]

    n_sne = 7000

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne_results = tsne.fit_transform(x_train_pca_to_plot)


    y_train_to_plot = y_train_real#[index_to_get]
    data_list = []
    unique_labels = np.unique(y_train_real)
    for label_value in unique_labels:
        labels_idx = np.where(y_train_to_plot == label_value)[0]
        data_list.append(tsne_pca_results[labels_idx])



    # Create plot
    fig = plt.figure()

    for label in range(len(data_list)):
        x = data_list[label][:, 0]
        y = data_list[label][:, 1]
        plt.scatter(x, y, alpha=0.8, edgecolors='none', label=label)
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()




