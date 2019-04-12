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

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

REAL_DATA_NAME = 'starlight_new_bal_1.00.pkl'


def targets_to_numbers(targets):
  target_keys = np.unique(targets)
  target_keys_idxs = np.argsort(np.unique(targets))
  targets_as_numbers = target_keys_idxs[
    np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]
  return targets_as_numbers


def get_data_from_set(set, magnitude_key, time_key):
  magnitudes = np.asarray(set[magnitude_key])
  time = np.asarray(set[time_key])
  x = np.stack((magnitudes, time), axis=-1)
  x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
  y = np.asarray(set['class'])
  x, y = shuffle(x, y, random_state=42)
  y = targets_to_numbers(y)
  return x, y


def read_data_irregular_sampling(file, magnitude_key='original_magnitude',
    time_key='time', verbose=False):
  dataset_partitions = np.load(file)
  if verbose:
    print(dataset_partitions[0].keys())
  x_train, y_train = get_data_from_set(dataset_partitions[0], magnitude_key,
                                       time_key)
  x_val, y_val = get_data_from_set(dataset_partitions[1], magnitude_key,
                                   time_key)
  x_test, y_test = get_data_from_set(dataset_partitions[2], magnitude_key,
                                     time_key)
  return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
  path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data',
                                   'datasets_original', 'REAL', REAL_DATA_NAME)
  x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
    read_data_irregular_sampling(
        path_to_real_data, magnitude_key='original_magnitude', time_key='time')
  x_train_real = x_train_real[:, :, 0, 0]
  scaler = StandardScaler()
  scaler.fit(x_train_real)
  x_train_scaled = x_train_real  # scaler.transform(x_train_real)

  pca = PCA()
  pca.fit(x_train_scaled)
  x_train_pca = pca.transform(x_train_scaled)

  variance_precentage = pca.explained_variance_ratio_
  cum_sum_variance = np.cumsum(variance_precentage)
  indx_important_pca_components = np.argmax(cum_sum_variance > 0.9)

  indexes_of_array = np.arange(x_train_pca.shape[0])
  np.random.shuffle(indexes_of_array)
  index_to_get_val = 5000
  index_to_get = indexes_of_array[:index_to_get_val]
  x_train_pca_to_plot = x_train_pca[:, :indx_important_pca_components][
    index_to_get]

  n_sne = 7000

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
  tsne_pca_results = tsne_results = tsne.fit_transform(x_train_pca_to_plot)

  y_train_to_plot = y_train_real[index_to_get]
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
