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
from sklearn.ensemble import RandomForestClassifier
import feature_extraction.tinkering_FATS as FATS_extractor
import matplotlib.pyplot as plt
from feature_extraction.rf_over_FATS import plot_confusion_matrix

REAL_DATA_NAME = 'catalina_north9classes.pkl'  # 'starlight_new_bal_0.10.pkl'#
SYN_DATA_NAME = 'catalina_amp_irregular_9classes_generated_10000.pkl'
REAL_FEATURES_FOLDER = os.path.join('TSTR_data', 'datasets_original', 'REAL', '9classes_100_100')
REAL_FEATURES_NAME = 'catalina_north9classes_features.pkl'
SYN_FEATURES_FOLDER = os.path.join('TSTR_data', 'generated', 'catalina_amp_irregular_9classes')
SYN_FEATURES_NAME = 'catalina_north9classes_features.pkl'
TEST_TYPE = 'FATS'


if __name__ == "__main__":
    path_to_real_data = os.path.join(PATH_TO_PROJECT, 'TSTR_data', 'datasets_original', 'REAL', REAL_DATA_NAME)
    path_to_syn_data = os.path.join(PATH_TO_PROJECT, SYN_FEATURES_FOLDER, SYN_DATA_NAME)

    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_data, magnitude_key='original_magnitude_random', time_key='time_random')
    x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_syn_data, magnitude_key='generated_magnitude', time_key='time')

    path_to_features = os.path.join(PATH_TO_PROJECT, SYN_FEATURES_FOLDER, SYN_FEATURES_NAME)
    features_syn = FATS_extractor.load_pickle(path_to_features)
    train_features = features_syn['train']
    path_to_real_features = os.path.join(PATH_TO_PROJECT, REAL_FEATURES_FOLDER, REAL_FEATURES_NAME)
    features_real = FATS_extractor.load_pickle(path_to_real_features)
    val_features = features_real['val']
    test_features = features_real['test']

    clf = RandomForestClassifier()
    #clf.fit(train_features, y_train_real)
    clf.fit(train_features, y_train_syn)
    print(clf.feature_importances_)
    y_predict_classes_test = clf.predict(test_features)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test_real, y_predict_classes_test)
    print('Accuracy Test conf %.4f' % (
            np.trace(confusion_matrix) / np.sum(confusion_matrix)))

    plot_confusion_matrix(
        cm=confusion_matrix, classes=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd', 'beta Lyrae', 'HADS'],
        n_class_val=0,
        title='%s Conf matrix for %i classes; Acc %.4f' % ('FATS_TSTR',
                                                           9,
                                                           np.trace(confusion_matrix) / np.sum(confusion_matrix)))
    plt.show()