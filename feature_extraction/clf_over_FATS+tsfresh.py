import sys
import os

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import feature_extraction.FATS_extractor as FATS_extractor
import matplotlib.pyplot as plt
from parameters import general_keys

TEST_TYPE_LIST = ['FATS+tsfresh_RF_TRTR', 'FATS+tsfresh_RF_TSTR', 'FATS+tsfresh_XGboost_TRTR',
             'FATS+tsfresh_XGboost_TSTR', 'FATS+tsfresh_LGBM_TRTR', 'FATS+tsfresh_LGBM_TSTR']
PATH_TO_SAVE_CONF_MAT = os.path.join('results', 'conf_mat', 'trees')
NAME_REAL_DATA = 'catalina_north9classes.pkl'
NAME_SYN_DATA = 'catalina_amp_irregular_9classes_generated_10000.pkl'
FOLDER_REAL_DATA = os.path.join('TSTR_data', 'datasets_original', 'REAL',
                                '9classes_100_100')
FOLDER_SYN_DATA = os.path.join('TSTR_data', 'generated',
                               'catalina_amp_irregular_9classes')
NAME_REAL_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'
NAME_REAL_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh.pkl'
NAME_SYN_TSFRESH_FEATURES = 'catalina_north9classes_features_tsfresh_concatenated.pkl'
NAME_SYN_FATS_FEATURES = 'catalina_north9classes_features_fats.pkl'



def check_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def plot_confusion_matrix(cm, classes, n_class_val,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues, path_to_save='', test_type=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
      if normalize:
        title = 'Normalized confusion matrix'
      else:
        title = 'Confusion matrix, without normalization'

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    check_dir(os.path.join(PATH_TO_PROJECT, path_to_save))
    if path_to_save != '':
      fig.savefig(
          os.path.join(PATH_TO_PROJECT, path_to_save,
                       '%s_conf_matrix_class%i.png' % (test_type, n_class_val)))


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
            merged_features[set_keys] = np.concatenate(merged_features[set_keys],
                                                       axis=-1)
    return merged_features


def train_clf_and_plot_conf_matrix(test_type):
    path_to_real_light_curves = os.path.join(PATH_TO_PROJECT, FOLDER_REAL_DATA,
                                             NAME_REAL_DATA)
    path_to_syn_light_curves = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA,
                                            NAME_SYN_DATA)

    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_real_light_curves, magnitude_key='original_magnitude_random',
            time_key='time_random')
    x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = \
        FATS_extractor.read_data_irregular_sampling(
            path_to_syn_light_curves, magnitude_key='generated_magnitude',
            time_key='time')

    path_to_real_fats_features = os.path.join(PATH_TO_PROJECT, FOLDER_REAL_DATA,
                                              NAME_REAL_FATS_FEATURES)
    path_to_real_tsfresh_features = os.path.join(PATH_TO_PROJECT,
                                                 FOLDER_REAL_DATA,
                                                 NAME_REAL_TSFRESH_FEATURES)
    path_to_syn_fats_features = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA,
                                             NAME_SYN_FATS_FEATURES)
    path_to_syn_tsfresh_features = os.path.join(PATH_TO_PROJECT, FOLDER_SYN_DATA,
                                                NAME_SYN_TSFRESH_FEATURES)

    real_merged_features = load_and_concatenate_features(
        [path_to_real_fats_features, path_to_real_tsfresh_features])
    syn_merged_features = load_and_concatenate_features(
        [path_to_syn_fats_features, path_to_syn_tsfresh_features])

    if 'TRTR' in test_type:
        train_features = real_merged_features[general_keys.TRAIN_SET_KEY]
        y_train = y_train_real
    elif 'TSTR' in test_type:
        train_features = syn_merged_features[general_keys.TRAIN_SET_KEY]
        y_train = y_train_syn
    val_features = real_merged_features[general_keys.VAL_SET_KEY]
    test_features = real_merged_features[general_keys.TEST_SET_KEY]

    if 'RF' in test_type:
      params = {'n_jobs': -1, 'n_estimators': 100, 'criterion': 'entropy',
                'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 2,
      'max_features': None}
      clf = RandomForestClassifier(**params)
    elif 'XGB' in test_type:
      params = {'n_jobs': -1, 'max_depth': 9, 'n_estimators': 200,
                'learning_rate': 0.1, 'subsample': 1.0,
                'colsample_bytree': 0.8, 'gamma': 0.5, 'min_child_weight': 5,
                'objective': 'multi:softprob'}
      clf = XGBClassifier(**params)
    elif 'LGB' in test_type:
      params = {'n_jobs': -1, 'max_depth': 9, 'n_estimators': 200,
      'learning_rate': 0.1, 'subsample': 1.0,
                'colsample_bytree': 0.8, 'gamma': 0.5, 'min_child_weight': 5,
                'objective': 'multi:softprob'}
      clf = LGBMClassifier(**params) #default params: #clf.get_params().keys()

    clf.fit(X=train_features, y=y_train)

    print('feature importances %s' % str(
        np.array(clf.feature_importances_) / np.sum(clf.feature_importances_)))
    y_predict_classes_test = clf.predict(test_features)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test_real,
                                                        y_predict_classes_test)
    print('Accuracy Test conf %.4f' % (
            np.trace(confusion_matrix) / np.sum(confusion_matrix)))

    plot_confusion_matrix(
        cm=confusion_matrix,
        classes=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd', 'beta Lyrae',
                 'HADS'],
        n_class_val=np.unique(y_train_real).shape[0],
        title='%s Conf matrix for %i classes; Acc %.4f' % (test_type,
                                                           np.unique(
                                                               y_train_real).shape[
                                                               0],
                                                           np.trace(
                                                               confusion_matrix) / np.sum(
                                                               confusion_matrix)),
         path_to_save=PATH_TO_SAVE_CONF_MAT, test_type=test_type)
    #plt.show()
    plt.close()
    return confusion_matrix


if __name__ == "__main__":
    for test_type in TEST_TYPE_LIST:
        train_clf_and_plot_conf_matrix(test_type)
