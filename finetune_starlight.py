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

DROP_OUT_RATE = 0.5
PATIENCE = 20
BN_CONDITION = 'batch_norm_'  # ''
BASE_REAL_NAME = 'starlight_noisy_irregular_all_same_set_amp_balanced_larger_train'
versions = ['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
RESULTS_NAME = 'finetune_small_lr_%sdp_%.1f_pt_%i_%s' % (
    BN_CONDITION, DROP_OUT_RATE, PATIENCE, BASE_REAL_NAME)
FOLDER_TO_SAVE_IN = 'fine_tune'
RUNS = 10

#from best 50-50 gan
AUGMENTED_OR_NOT_EXTRA_STR = '_augmented_50-50'
BEST_GAN_NAME = 'trts_%sdp_%.1f_pt_%i_%s_%s' % (
BN_CONDITION, DROP_OUT_RATE, PATIENCE, AUGMENTED_OR_NOT_EXTRA_STR, BASE_REAL_NAME)
SET_KEY_FOR_BEST_METRIC = 'training'
BEST_METRIC_KEY = 'VAL_ACC'

PATIENCE = 30
PATIENCE_FINE = 200


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date = '2803'


def main(result_dict={}, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE=1.0, v=''):
    real_data_folder = os.path.join('datasets_original', 'REAL')
    dataset_real_pkl = '%s%.2f.pkl' % (BASE_REAL_NAME, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    syn_data_name = os.path.join('%s%s%.2f' % (BASE_REAL_NAME, v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE))

    PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY = str(PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY] = {'training': {}, 'testing': {}}
    print("REAL Training set to load %s" % dataset_real_pkl)
    print("SYN Training set to load %s" % syn_data_name)

    def read_data(file):

        with open(file, 'rb') as f: data = pickle.load(f)

        X_train = np.asarray(data[0]['generated_magnitude'])
        # print(X_train.shape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
        # print(X_train.shape)
        y_train = np.asarray(data[0]['class'])
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        y_train = change_classes(y_train)
        y_train = to_categorical(y_train)

        X_val = np.asarray(data[1]['generated_magnitude'])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
        y_val = np.asarray(data[1]['class'])
        y_val = change_classes(y_val)
        y_val = to_categorical(y_val)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        X_test = np.asarray(data[2]['generated_magnitude'])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
        y_test = np.asarray(data[2]['class'])
        y_test = change_classes(y_test)
        y_test = to_categorical(y_test)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def read_data_original_irr(file):

        with open(file, 'rb') as f: data = pickle.load(f)

        print(data[0].keys())

        mgt = np.asarray(data[0]['original_magnitude'])
        t = np.asarray(data[0]['time'])
        X_train = np.stack((mgt, t), axis=-1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])

        y_train = np.asarray(data[0]['class'])

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        y_train = change_classes(y_train)
        y_train = to_categorical(y_train)

        mgt = np.asarray(data[1]['original_magnitude'])
        t = np.asarray(data[1]['time'])
        X_val = np.stack((mgt, t), axis=-1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
        y_val = np.asarray(data[1]['class'])
        y_val = change_classes(y_val)
        y_val = to_categorical(y_val)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        mgt = np.asarray(data[2]['original_magnitude'])
        t = np.asarray(data[2]['time'])
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
    irr = True
    dataset_syn_pkl = syn_data_name + '_generated.pkl'
    one_d = False

    ## Train on synthetic

    X_train_syn, y_train_syn, X_val_syn, y_val_syn, X_test_syn, y_test_syn = read_data_generated_irr(
        os.path.join('TSTR_data', 'generated', syn_data_name, dataset_syn_pkl))
    X_train_real, y_train_real, X_val_real, y_val_real, X_test_real, y_test_real = read_data_original_irr(
        os.path.join('TSTR_data', real_data_folder, dataset_real_pkl))


    print('')
    print('Training new model')
    print('')

    batch_size = 512
    epochs = 200

    num_classes = 3

    m = Model_(batch_size, 100, num_classes, drop_rate=DROP_OUT_RATE)

    if BN_CONDITION == 'batch_norm_':
        model = m.cnn2_batch()
    else:
        model = m.cnn2()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## callbacks
    history = my_callbacks.Histories()

    checkpoint = ModelCheckpoint('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainonsynthetic.hdf5',
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.00000001, patience=PATIENCE, verbose=1, mode='max')

    model.fit(X_train_syn, y_train_syn, epochs=epochs, batch_size=batch_size, validation_data=(X_val_real, y_val_real),
              callbacks=[history,
                         checkpoint,
                         earlyStopping  # ,
                         # rocauc,
                         # inception
                         ])

    model = load_model('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainonsynthetic.hdf5')

    print('Training metrics:')

    score_train = model.evaluate(X_train_syn, y_train_syn, verbose=1)
    score_val = model.evaluate(X_val_real, y_val_real, verbose=1)

    print('ACC : ', score_train[1])
    print('VAL_ACC : ', score_val[1])
    print('LOSS : ', score_train[0])
    print('VAL_LOSS : ', score_val[0])



    #fine tunning
    K.set_value(model.optimizer.lr, 0.00001)

    checkpoint = ModelCheckpoint('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainfinetune.hdf5',
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.00000001, patience=PATIENCE_FINE, verbose=1, mode='max')
    model.fit(X_train_real, y_train_real, epochs=epochs, batch_size=batch_size, validation_data=(X_val_real, y_val_real),
              callbacks=[history,
                         checkpoint,
                         earlyStopping  # ,
                         # rocauc,
                         # inception
                         ])

    model = load_model('TSTR_' + date + '/train/' + syn_data_name + '/weights.best.trainfinetune.hdf5')

    ## Test on real

    score_val = model.evaluate(X_val_real, y_val_real, verbose=1)

    print('fine tune VAL_ACC : ', score_val[1])
    print('fine tune VAL_LOSS : ', score_val[0])


    print('\nTest metrics:')
    print('\nTest on real:')

    score = model.evaluate(X_test_real, y_test_real, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing'] = {
        'test loss on real': score[0], 'Test accuracy on real': score[1]  # , 'auc roc on real': roc
    }

    ## Test on syn

    print('\nTest on synthetic:')

    score = model.evaluate(X_test_syn, y_test_syn, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['training'] = {
        'VAL_ACC': score_val[1], 'TRAIN_ACC': score_train[1],
        'TRAIN_LOSS': score_train[0], 'VAL_LOSS': score_val[0]
    }

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing']['test loss on syn'] = score[0]
    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing']['Test accuracy on syn'] = score[1]

    keras.backend.clear_session()
    del model



def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    best_gans_dict = np.load(os.path.join('results', 'select_best_gan',
                                          'best_gan_dict_' + str(RUNS) + '_runs' + BEST_GAN_NAME + '_'.join(
                                              versions) + '.pkl'))
    result_dict_for_different_versions_runs = {}
    for run_idx in range(RUNS):
        result_dict_for_different_versions_runs['run_%i' % run_idx] = {}
    for run_idx in result_dict_for_different_versions_runs.keys():
        dict_single_version = result_dict_for_different_versions_runs[run_idx]
        MIN_LIM = 10
        MAX_LIM = 100
        keep_samples_list = np.round(np.logspace(np.log10(MIN_LIM), np.log10(MAX_LIM), num=6)) / 100
        for keep_sample in keep_samples_list:
            print('loading best gan for %s keep %s version %s acc %s' % (run_idx,
                                                                         str(keep_sample),
                                                                         best_gans_dict[str(keep_sample)][
                                                                             'best_version'],
                                                                         str(best_gans_dict[str(keep_sample)][
                                                                                 'mean_%s' % BEST_METRIC_KEY])))
            best_gan_for_percentage = best_gans_dict[str(keep_sample)]['best_version']
            main(dict_single_version, keep_sample, best_gan_for_percentage)
        print(dict_single_version)
    print(result_dict_for_different_versions_runs)

    check_dir(os.path.join('results', FOLDER_TO_SAVE_IN))
    pickle.dump(result_dict_for_different_versions_runs, open(
        os.path.join('results', FOLDER_TO_SAVE_IN, 'single_gan_results' + RESULTS_NAME + '_'.join(versions) + '.pkl'),
        "wb"))

