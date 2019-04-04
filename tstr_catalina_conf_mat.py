from model_keras import *
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import pickle
import my_callbacks
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score, accuracy_score
import keras
import keras.backend as K
import sklearn

# from sklearn.metrics import confusion_matrix

DROP_OUT_RATE = 0.5
PATIENCE = 30
BN_CONDITION = 'batch_norm_'  # ''
BASE_GEN_DATA_FOLDER_NAME = 'catalina_amp_irregular_'
BASE_REAL_NAME = 'catalina_north'
TEST_TYPE = 'tstr'
RESULTS_NAME = '%s_for_conf_%s' % ( TEST_TYPE,
    BASE_GEN_DATA_FOLDER_NAME)
FOLDER_TO_SAVE_IN = os.path.join('conf_mat', TEST_TYPE, 'catalina')
# RUNS = 10
EARLY_STOP_ON = 'val_acc'
EARLY_STOP_ON_COD = 'max'
CLASSES_TO_RUN = np.arange(2, 10, 1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date = '2803'
ORIGNAL_MAG_KEY = 'original_magnitude_random'
ORIGINAL_TIME_KEY = 'time_random'


def main(result_dict={}, catalina_n_classes=1):
    real_data_folder = os.path.join('datasets_original', 'REAL', '%iclasses_100_100' % catalina_n_classes)
    dataset_real_pkl = '%s%iclasses.pkl' % (BASE_REAL_NAME, catalina_n_classes)
    syn_data_name = os.path.join('%s%iclasses' % (BASE_GEN_DATA_FOLDER_NAME, catalina_n_classes))

    catalina_n_classes_str = catalina_n_classes#str(catalina_n_classes)
    result_dict[catalina_n_classes_str] = {'training': {}, 'testing': {}}
    #result_dict = {'training': {}, 'testing': {}}
    print("\nREAL Training set to load %s\n" % dataset_real_pkl)

    # print("SYN Training set to load %s" % syn_data_name)

    def read_data_original_irr(file):

        with open(file, 'rb') as f: data = pickle.load(f)

        print(data[0].keys())

        mgt = np.asarray(data[0][ORIGNAL_MAG_KEY])
        t = np.asarray(data[0][ORIGINAL_TIME_KEY])
        X_train = np.stack((mgt, t), axis=-1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])

        y_train = np.asarray(data[0]['class'])

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        y_train = change_classes(y_train)
        y_train = to_categorical(y_train)

        mgt = np.asarray(data[1][ORIGNAL_MAG_KEY])
        t = np.asarray(data[1][ORIGINAL_TIME_KEY])
        X_val = np.stack((mgt, t), axis=-1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
        y_val = np.asarray(data[1]['class'])
        y_val = change_classes(y_val)
        y_val = to_categorical(y_val)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        mgt = np.asarray(data[2][ORIGNAL_MAG_KEY])
        t = np.asarray(data[2][ORIGINAL_TIME_KEY])
        X_test = np.stack((mgt, t), axis=-1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
        y_test = np.asarray(data[2]['class'])
        y_test = change_classes(y_test)
        y_test = to_categorical(y_test)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def read_data_generated_irr(file):

        with open(file, 'rb') as f: data = pickle.load(f)

        print(data[0].keys())

        mgt = np.asarray(data[0]['generated_magnitude'])
        t = np.asarray(data[0]['time'])
        X_train = np.stack((mgt, t), axis=-1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
        # print(X_train.shape)
        y_train = np.asarray(data[0]['class'])
        print(np.unique(y_train))
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        #  for i in y_train:
        #     if i != None:
        #        print(i)
        y_train = change_classes(y_train)
        y_train = to_categorical(y_train)

        mgt = np.asarray(data[1]['generated_magnitude'])
        t = np.asarray(data[1]['time'])
        X_val = np.stack((mgt, t), axis=-1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
        y_val = np.asarray(data[1]['class'])
        y_val = change_classes(y_val)
        y_val = to_categorical(y_val)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        mgt = np.asarray(data[2]['generated_magnitude'])
        t = np.asarray(data[2]['time'])
        X_test = np.stack((mgt, t), axis=-1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
        y_test = np.asarray(data[2]['class'])
        y_test = change_classes(y_test)
        y_test = to_categorical(y_test)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def change_classes(targets):
        # print(targets)
        target_keys = np.unique(targets)
        # print(target_keys)
        target_keys_idxs = np.argsort(np.unique(targets))
        targets = target_keys_idxs[np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]

        return targets

    def open_data(file):

        with open(file, 'rb') as f: data = pickle.load(f)

        print(len(data['generated_magnitude']))

        X = np.asarray(data['generated_magnitude'])
        X = X.reshape(X.shape[0], X.shape[1], 1, 1)
        y = np.asarray(data['class'])
        X, y = shuffle(X, y, random_state=42)
        y = change_classes(y)
        y = to_categorical(y)

        return X, y

    def check_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    check_dir('TSTR_' + date)
    check_dir('TSTR_' + date + '/train/')
    check_dir('TSTR_' + date + '/train/')
    check_dir('TSTR_' + date + '/train/' + syn_data_name)
    check_dir('TSTR_' + date + '/test/')
    check_dir('TSTR_' + date + '/test/' + syn_data_name)

    # if else
    # irr = True
    dataset_syn_pkl = syn_data_name + '_generated.pkl'
    # one_d = False

    ## Train on real

    X_train_syn, y_train_syn, X_val_syn, y_val_syn, X_test_syn, y_test_syn = read_data_generated_irr(
         os.path.join('TSTR_data', 'generated', syn_data_name, dataset_syn_pkl))
    X_train_real, y_train_real, X_val_real, y_val_real, X_test_real, y_test_real = read_data_original_irr(
        os.path.join('TSTR_data', real_data_folder, dataset_real_pkl))

    print('')
    print('Training new model')
    print('')

    batch_size = 512
    epochs = 10000

    num_classes = catalina_n_classes

    m = Model_(batch_size, 100, num_classes, drop_rate=DROP_OUT_RATE)

    if BN_CONDITION == 'batch_norm_':
        model = m.cnn2_batch()
    else:
        model = m.cnn2()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## callbacks
    history = my_callbacks.Histories()

    checkpoint = ModelCheckpoint('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainonsyn.hdf5',
                                 monitor=EARLY_STOP_ON, verbose=1, save_best_only=True, mode=EARLY_STOP_ON_COD)
    earlyStopping = EarlyStopping(monitor=EARLY_STOP_ON, min_delta=0.00000001, patience=PATIENCE, verbose=1,
                                  mode=EARLY_STOP_ON_COD)

    model.fit(X_train_syn, y_train_syn, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val_real, y_val_real),
              callbacks=[history,
                         checkpoint,
                         earlyStopping  # ,
                         # rocauc,
                         # inception
                         ])

    model = load_model('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainonsyn.hdf5')

    print('Training metrics:')

    score_train = model.evaluate(X_train_syn, y_train_syn, verbose=1)
    score_val = model.evaluate(X_val_real, y_val_real, verbose=1)

    print('ACC : ', score_train[1])
    print('VAL_ACC : ', score_val[1])
    print('LOSS : ', score_train[0])
    print('VAL_LOSS : ', score_val[0])

    # fine tunning
    # K.set_value(model.optimizer.lr, 0.00005)
    #
    # checkpoint = ModelCheckpoint('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainfinetune.hdf5',
    #                              monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.00000001, patience=PATIENCE_FINE, verbose=1, mode='max')
    # model.fit(X_train_real, y_train_real, epochs=epochs, batch_size=batch_size, validation_data=(X_val_real, y_val_real),
    #           callbacks=[history,
    #                      checkpoint,
    #                      earlyStopping  # ,
    #                      # rocauc,
    #                      # inception
    #                      ])
    #
    # model = load_model('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainfinetune.hdf5')

    ## Test on real

    # score_val = model.evaluate(X_val_real, y_val_real, verbose=1)
    #
    # print('fine tune VAL_ACC : ', score_val[1])
    # print('fine tune VAL_LOSS : ', score_val[0])

    print('\nTest metrics:')
    print('\nTest on real:')

    score_test = model.evaluate(X_test_real, y_test_real, verbose=1)
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])

    result_dict[catalina_n_classes_str]['testing'] = {
        'test loss on real': score_test[0], 'Test accuracy on real': score_test[1]  # , 'auc roc on real': roc
    }

    ## Test on syn

    print('\nTest on synthetic:')

    # score = model.evaluate(X_test_syn, y_test_syn, verbose=1)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    result_dict[catalina_n_classes_str]['training'] = {
        'VAL_ACC': score_val[1], 'TRAIN_ACC': score_train[1],
        'TRAIN_LOSS': score_train[0], 'VAL_LOSS': score_val[0]
    }

    # result_dict[catalina_n_classes_str]['testing']['test loss on syn'] = score[0]
    # result_dict[catalina_n_classes_str]['testing']['Test accuracy on syn'] = score[1]

    y_predict_prob_test = model.predict(X_test_real)
    y_predict_classes_test = y_predict_prob_test.argmax(axis=-1)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test_real.argmax(axis=-1), y_predict_classes_test)
    print('Accuracy Test conf %.4f, accuracy eval %.4f' % (np.trace(confusion_matrix)/np.sum(confusion_matrix), score_test[1]))

    keras.backend.clear_session()
    del model
    return confusion_matrix


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


if __name__ == '__main__':
    result_dict_for_different_n_classes = {}
    for n_class_value in CLASSES_TO_RUN:
        result_dict_for_different_n_classes['class%i' % n_class_value] = {}
    for n_class_value in CLASSES_TO_RUN:
        dict_single_class_results = result_dict_for_different_n_classes['class%i' % n_class_value]

        conf_matrix = main(dict_single_class_results, n_class_value)
        check_dir(os.path.join('results', FOLDER_TO_SAVE_IN, 'class%i' % n_class_value))

        plot_confusion_matrix(
            cm=conf_matrix, classes=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd', 'beta Lyrae', 'HADS'],
            n_class_val=n_class_value,
            title='%s Conf matrix for %i classes; Acc %.4f' % ( TEST_TYPE,
                n_class_value, dict_single_class_results[n_class_value]['testing']['Test accuracy on real']),
            path_to_save=os.path.join('results', FOLDER_TO_SAVE_IN, 'class%i' % n_class_value))

        plot_confusion_matrix(
            cm=conf_matrix, classes=['EW', 'RRc', 'EA', 'RRab', 'RS CVn', 'LPV', 'RRd', 'beta Lyrae', 'HADS'],
            n_class_val=n_class_value,
            title='%s Conf matrix for %i classes; Acc %.4f' % ( TEST_TYPE,
                n_class_value, dict_single_class_results[n_class_value]['testing']['Test accuracy on real']),
            path_to_save=os.path.join('results', FOLDER_TO_SAVE_IN))

        pickle.dump(dict_single_class_results, open(
            os.path.join('results', FOLDER_TO_SAVE_IN, 'class%i' % n_class_value,
                         RESULTS_NAME + '_class%i' % n_class_value + '.pkl'), "wb"))

    pickle.dump(result_dict_for_different_n_classes, open(
        os.path.join('results', FOLDER_TO_SAVE_IN,
                     RESULTS_NAME + 'all_classes' + '.pkl'), "wb"))
