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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import feature_extraction.tinkering_FATS as FATS_extractor
import matplotlib.pyplot as plt
from feature_extraction.rf_over_FATS import plot_confusion_matrix
from parameters import general_keys

NAME_REAL_DATA = 'catalina_north9classes.pkl'
NAME_SYN_DATA = 'catalina_amp_irregular_9classes_generated_10000.pkl'
FOLDER_REAL_DATA = os.path.join('TSTR_data', 'datasets_original', 'REAL', '9classes_100_100')
FOLDER_SYN_DATA = os.path.join('TSTR_data', 'generated', 'catalina_amp_irregular_9classes')
NAME_REAL_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'
NAME_REAL_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh.pkl'
NAME_SYN_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh_concatenated.pkl'
NAME_SYN_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'
TEST_TYPE = 'FATS_TSTR'


def load_multiple_sets(path_list):
    sets = []
    for path in path_list:
        sets.append(FATS_extractor.load_pickle(path))
    return sets


def load_and_concatenate_features(path_list):
    merged_features = {}
    feature_sets = load_multiple_sets(path_list)
    for set_keys in feature_sets[0].keys():
        merged_features[set_keys] = []
        for single_feature_set in feature_sets:
            merged_features[set_keys].append(single_feature_set[set_keys])
        if not isinstance(merged_features[set_keys][0], int):
            merged_features[set_keys] = np.concatenate(merged_features[set_keys], axis=-1)
    return merged_features

#def train_clf_and_plot_conf_matrix(folder_features_train, features_name):



if __name__ == "__main__":
    path_to_real_light_curves = os.path.join(PATH_TO_PROJECT, FOLDER_REAL_DATA, NAME_REAL_DATA)
    path_to_syn_light_curves = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA, NAME_SYN_DATA)

    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_light_curves, magnitude_key='original_magnitude_random', time_key='time_random')
    x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_syn_light_curves, magnitude_key='generated_magnitude', time_key='time')

    path_to_real_fats_features = os.path.join(PATH_TO_PROJECT, FOLDER_REAL_DATA, NAME_REAL_FATS_FEATURES)
    path_to_real_tsfresh_features = os.path.join(PATH_TO_PROJECT, FOLDER_REAL_DATA, NAME_REAL_TSFRESH_FEATURES)
    path_to_syn_fats_features = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA, NAME_SYN_FATS_FEATURES)
    path_to_syn_tsfresh_features = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA, NAME_SYN_TSFRESH_FEATURES)

    # real_features_fats = FATS_extractor.load_pickle(path_to_real_fats_features)
    # real_features_tsfresh = FATS_extractor.load_pickle(path_to_real_tsfresh_features)
    real_merged_features = load_and_concatenate_features([path_to_real_fats_features, path_to_real_tsfresh_features])
    syn_merged_features = load_and_concatenate_features([path_to_syn_fats_features, path_to_syn_tsfresh_features])

    train_features = real_merged_features[general_keys.TRAIN_SET_KEY]
    y_train = y_train_real
    val_features = real_merged_features[general_keys.VAL_SET_KEY]
    test_features = real_merged_features[general_keys.TEST_SET_KEY]

    # # params = {'n_jobs': -1, 'n_estimators': 100, 'criterion': 'entropy',
    # #           'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 2, 'max_features': None}
    # # clf = RandomForestClassifier(**params)
    #
    params = {'n_jobs': -1, 'max_depth': 9, 'n_estimators': 200, 'learning_rate': 0.1, 'subsample': 1.0,
              'colsample_bytree': 0.8, 'gamma': 0.5, 'min_child_weight': 5, 'objective': 'multi:softprob'}
    clf = XGBClassifier(**params)
    clf.fit(X=train_features, y=y_train)

    # params = {'n_jobs': -1, 'max_depth': 9, 'n_estimators': 200, 'learning_rate': 0.1, 'subsample': 1.0,
    #           'colsample_bytree': 0.8, 'gamma': 0.5, 'min_child_weight': 5, 'objective': 'multi:softprob'}
    # clf = LGBMClassifier(**params) #default params: #clf.get_params().keys()
    # clf.fit(X=train_features, y=y_train)

    print('feature importances %s' % str(np.array(clf.feature_importances_)/np.sum(clf.feature_importances_)))
    y_predict_classes_test = clf.predict(test_features)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test_real, y_predict_classes_test)
    print('Accuracy Test conf %.4f' % (
            np.trace(confusion_matrix) / np.sum(confusion_matrix)))

    plot_confusion_matrix(
        cm=confusion_matrix, classes=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd', 'beta Lyrae', 'HADS'],
        n_class_val=0,
        title='%s Conf matrix for %i classes; Acc %.4f' % (TEST_TYPE,
                                                           np.unique(y_train_real).shape[0],
                                                           np.trace(confusion_matrix) / np.sum(confusion_matrix)))
    plt.show()
