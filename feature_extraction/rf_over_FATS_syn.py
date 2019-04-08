import sys
import os
import FATS
import numpy as np
import pickle as pkl
import sklearn
import time
import datetime
from sklearn.ensemble import RandomForestClassifier
import feature_extraction.tinkering_FATS as FATS_extractor
import matplotlib.pyplot as plt

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

REAL_DATA_NAME = 'catalina_north9classes.pkl'  # 'starlight_new_bal_0.10.pkl'#
SYN_DATA_NAME = 'catalina_amp_irregular_9classes_generated_10000.pkl'
REAL_FEATURES_FOLDER = os.path.join('TSTR_data', 'datasets_original', 'REAL', '9classes_100_100')
REAL_FEATURES_NAME = 'catalina_north9classes_features.pkl'
SYN_FEATURES_FOLDER = os.path.join('TSTR_data', 'generated', 'catalina_amp_irregular_9classes')
SYN_FEATURES_NAME = 'catalina_north9classes_features.pkl'
TEST_TYPE = 'FATS'


def plot_confusion_matrix(cm, classes, n_class_val,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, path_to_save=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
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
    if path_to_save != '':
        fig.savefig(
            os.path.join(path_to_save, '%s_conf_matrix_class%i.png' % (TEST_TYPE, n_class_val)))


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