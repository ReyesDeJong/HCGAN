import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from modules.data_loaders.data_loader import DataLoader
import parameters.general_keys as general_keys

REAL_DATA_NAME = 'starlight_new_bal_1.00.pkl'

if __name__ == '__main__':
  path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data',
                                   'datasets_original', 'REAL', REAL_DATA_NAME)
  data_loader = DataLoader(
      magnitude_key=general_keys.ORIGINAL_MAGNITUDE, time_key=general_keys.TIME,
      data_path=path_to_real_data)
  x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
    data_loader.get_all_sets_data()
  x_train_real = x_train_real[:, :, 0]
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
