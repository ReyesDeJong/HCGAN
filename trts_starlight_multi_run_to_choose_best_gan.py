from model_keras import *
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import pickle
import my_callbacks
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import shutil
from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score, accuracy_score
import keras

DROP_OUT_RATE = 1.0
PATIENCE = 20
BN_CONDITION = 'batch_norm_'  # ''
BASE_REAL_NAME = 'starlight_noisy_irregular_all_same_set_amp_balanced_larger_train'
AUGMENTED_OR_NOT_EXTRA_STR = '_augmented_50-50'  # ''##
versions = ['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
RUNS = 10
RESULTS_NAME = 'trts_%sdp_%.1f_pt_%i_%s_%s' % (
BN_CONDITION, DROP_OUT_RATE, PATIENCE, AUGMENTED_OR_NOT_EXTRA_STR, BASE_REAL_NAME)
FOLDER_TO_SAVE_IN = 'select_best_gan'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date = '2803'
SET_KEY_FOR_BEST_METRIC = 'training'
BEST_METRIC_KEY = 'VAL_ACC'


def main(result_dict={}, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE=1.0, v=''):
    folder = '%s%s%.2f' % (BASE_REAL_NAME, v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    if AUGMENTED_OR_NOT_EXTRA_STR == '':
        in_TSTR_FOLDER = 'datasets_original/REAL/'
        dataset_real = '%s%s%s%.2f' % (
        BASE_REAL_NAME, AUGMENTED_OR_NOT_EXTRA_STR, '', PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    else:
        in_TSTR_FOLDER = 'augmented/'
        dataset_real = '%s%s%s%.2f' % (
        BASE_REAL_NAME, AUGMENTED_OR_NOT_EXTRA_STR, v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    #folder = dataset_real
    # folder = 'starlight_amp_noisy_irregular_all_%s%.2f' % (v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    # dataset_real = 'starlight_noisy_irregular_all_%s%.2f' % (v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    # same_set
    # folder = 'starlight_noisy_irregular_all_same_set_%s%.2f' % (v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    # dataset_real = 'starlight_noisy_irregular_all_same_set_%.2f' % PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE
    # for augmented
    # dataset_real = 'starlight_random_sample_augmented_%s%.2f' % (v, PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)

    PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY = str(PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE)
    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY] = {'training': {}, 'testing': {}}

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
        # print(X_train.shape)
        # print(X_train.T.shape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
        # print(X_train.shape)
        y_train = np.asarray(data[0]['class'])
        # print(np.unique(y_train))
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
        # print(X_train.shape)
        # print(X_train.T.shape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
        # print(X_train.shape)
        y_train = np.asarray(data[0]['class'])
        # print(np.unique(y_train))
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        #	for i in y_train:
        #		if i != None:
        #			print(i)
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

    def evaluation(X_test, y_test, n_classes):
        y_pred_prob = model.predict_proba(X_test)

        n = 10
        probs = np.array_split(y_pred_prob, n)

        score = []
        mean = []
        std = []

        Y = []
        for prob in probs:
            ys = np.zeros(n_classes)  # [0, 0
            for class_i in range(n_classes):
                for j in prob:
                    ys[class_i] = ys[class_i] + j[class_i]

            ys[:] = [x / len(prob) for x in ys]

            Y.append(np.asarray(ys))

        ep = 1e-12
        tmp = []
        for s in range(n):
            kl = (probs[s] * np.log((probs[s] + ep) / Y[s])).sum(axis=1)
            E = np.mean(kl)
            IS = np.exp(E)
            # pdb.set_trace()
            tmp.append(IS)

        score.append(tmp)
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))

        print('Inception Score:\nMean score : ', mean[-1])
        print('Std : ', std[-1])

        return score, mean, std

    def check_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    check_dir('TRTS_' + date)
    check_dir('TRTS_' + date + '/train/')
    check_dir('TRTS_' + date + '/train/')
    check_dir('TRTS_' + date + '/train/' + folder)
    check_dir('TRTS_' + date + '/test/')
    check_dir('TRTS_' + date + '/test/' + folder)

    # if os.path.isfile('TRTS_' + date + '/train/' + folder + '/train_model.h5'):
    #    os.remove('TRTS_' + date + '/train/' + folder + '/train_model.h5')
    #    shutil.rmtree('TRTS_' + date + '/test/' + folder)

    # else:

    irr = True
    one_d = False

    ## Train on real

    # dataset_real = 'catalina_random_full_north_9classes'
    if irr == True:
        X_train, y_train, X_val, y_val, X_test, y_test = read_data_original_irr(
            'TSTR_data/' + in_TSTR_FOLDER + dataset_real + '.pkl')  # datasets_original/REAL/' + dataset_real + '.pkl')
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = read_data(
            'TSTR_data/' + in_TSTR_FOLDER + dataset_real + '.pkl')

    print('')
    print('Training new model')
    print('')

    batch_size = 512
    epochs = 200

    num_classes = 3

    m = Model_(batch_size, 100, num_classes, drop_rate=DROP_OUT_RATE)

    # if one_d == True:
    #    model = m.cnn()
    # else:
    #    model = m.cnn2()
    if BN_CONDITION == 'batch_norm_':
        model = m.cnn2_batch()
    else:
        model = m.cnn2()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## callbacks
    history = my_callbacks.Histories()
    # rocauc = my_callbacks.ROC_AUC(X_train, y_train, X_test, y_test)
    # inception = my_callbacks.Inception(X_test, num_classes)

    checkpoint = ModelCheckpoint('TRTS_' + date + '/train/' + folder + '/weights.best.train.hdf5', monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.00000001, patience=PATIENCE, verbose=1, mode='max')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
              callbacks=[history,
                         checkpoint,
                         earlyStopping  # ,
                         # rocauc,
                         # inception
                         ])

    model = load_model('TRTS_' + date + '/train/' + folder + '/weights.best.train.hdf5')
    os.remove('TRTS_' + date + '/train/' + folder + '/weights.best.train.hdf5')

    # Create dictionary, then save into two different documments.
    ## Loss
    history_dictionary_loss = history.loss
    np.save('TRTS_' + date + '/train/' + folder + '/train_history_loss.npy', history_dictionary_loss)
    ## Val Loss
    history_dictionary_val_loss = history.val_loss
    np.save('TRTS_' + date + '/train/' + folder + '/train_history_val_loss.npy', history_dictionary_val_loss)
    ## Acc
    history_dictionary_acc = history.acc
    np.save('TRTS_' + date + '/train/' + folder + '/train_history_acc.npy', history_dictionary_acc)
    ## Val Acc
    history_dictionary_val_acc = history.val_acc
    np.save('TRTS_' + date + '/train/' + folder + '/train_history_val_acc.npy', history_dictionary_val_acc)
    ## AUC ROC
    # roc_auc_dictionary = rocauc.roc_auc
    # np.save('TRTS_' + date + '/train/' + folder + '/train_rocauc_dict.npy', roc_auc_dictionary)
    ## IS
    # scores_dict = inception.score
    # np.save('TRTS_' + date + '/train/' + folder + '/train_is.npy', scores_dict)
    # mean_scores_dict = inception.mean
    # np.save('TRTS_' + date + '/train/' + folder + '/train_is_mean.npy', mean_scores_dict)
    # std_scores_dict = inception.std
    # np.save('TRTS_' + date + '/train/' + folder + '/train_is_std.npy', std_scores_dict)

    ### plot loss and validation_loss v/s epochs
    plt.figure(1)
    plt.yscale("log")
    plt.plot(history.loss)
    plt.plot(history.val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('TRTS_' + date + '/train/' + folder + '/train_loss.png')
    ### plot acc and validation acc v/s epochs
    plt.figure(2)
    plt.yscale("log")
    plt.plot(history.acc)
    plt.plot(history.val_acc)
    plt.title('model acc')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('TRTS_' + date + '/train/' + folder + '/train_acc.png')

    print('Training metrics:')
    # print('Inception Score:\nMean score : ', mean_scores_dict[-1])
    # print('Std : ', std_scores_dict[-1])

    # model = load_model('TRTS_' + date + '/train/' + folder + '/weights.best.train.hdf5')

    score_train = model.evaluate(X_train, y_train, verbose=1)
    score_val = model.evaluate(X_val, y_val, verbose=1)

    print('ACC : ', score_train[1])
    print('VAL_ACC : ', score_val[1])
    print('LOSS : ', score_train[0])
    print('VAL_LOSS : ', score_val[0])

    ## Test on synthetic

    print('\nTest metrics:')
    print('\nTest on real:')

    dataset_syn = folder + '_generated'

    # sc, me, st = evaluation(X_test, y_test, num_classes)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onreal_is.npy', sc)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onreal_is_mean.npy', me)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onreal_is_std.npy', st)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    np.save('TRTS_' + date + '/test/' + folder + '/test_onreal_score.npy', score)

    # y_pred = model.predict(X_test)
    # roc = roc_auc_score(y_test, y_pred)
    # print('auc roc', roc)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onreal_rocauc.npy', roc)

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing'] = {
        'test loss on real': score[0], 'Test accuracy on real': score[1]  # , 'auc roc on real': roc
    }

    print('\nTest on synthetic:')
    if irr == True:
        X_train2, y_train2, X_val2, y_val2, X_test2, y_test2 = read_data_generated_irr(
            'TSTR_data/generated/' + folder + '/' + dataset_syn + '.pkl')
    else:
        X_train2, y_train2, X_val2, y_val2, X_test2, y_test2 = read_data(
            'TSTR_data/generated/' + folder + '/' + dataset_syn + '.pkl')

    # sc, me, st = evaluation(X_test2, y_test2, num_classes)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onsyn_is.npy', sc)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onsyn_is_mean.npy', me)
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onsyn_is_std.npy', st)

    score = model.evaluate(X_test2, y_test2, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # np.save('TRTS_' + date + '/test/' + folder + '/test_onsyn_score.npy', score)

    # y_pred = model.predict(X_test2)
    # roc = roc_auc_score(y_test2, y_pred)
    # print('auc roc', roc)

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['training'] = {
        #   'IS Mean': mean_scores_dict[-1],
        #   'IS Std': std_scores_dict[-1], 'ACC': np.mean(history_dictionary_acc),
        'VAL_ACC': score_val[1], 'TRAIN_ACC': score_train[1],
        'TRAIN_LOSS': score_train[0], 'VAL_LOSS': score_val[0]
    }

    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing']['test loss on syn'] = score[0]
    result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing']['Test accuracy on syn'] = score[1]
    # result_dict[PERCENTAGE_OF_SAMPLES_TO_KEEP_FOR_DISBALANCE_KEY]['testing']['auc roc on syn'] = roc
    # np.save('TRTS_' + date + '/test/' + folder + '/test_onsyn_rocauc.npy', roc)
    keras.backend.clear_session()
    del model


def get_percentage_version_mean_metric_dict(results_dict, set_key, metric_key):
    runs_keys = list(results_dict.keys())
    versions_keys = list(results_dict[runs_keys[0]].keys())
    percentage_keys = list(results_dict[runs_keys[0]][versions_keys[0]].keys())
    mean_metrics_dict = {}
    # create empty dict
    for percentage in percentage_keys:
        mean_metrics_dict[percentage] = {}
        for version in versions_keys:
            mean_metrics_dict[percentage][version] = {}
            for run in runs_keys:
                mean_metrics_dict[percentage][version][run] = None
    # fill dict
    for run in runs_keys:
        for version in versions_keys:
            for percentage in percentage_keys:
                mean_metrics_dict[percentage][version][run] = results_dict[run][version][percentage][set_key][
                    metric_key]

    # generate means
    for percentage in percentage_keys:
        for version in versions_keys:
            metric_list = []
            for run in runs_keys:
                metric_list.append(mean_metrics_dict[percentage][version][run])
            mean_metrics_dict[percentage][version]['mean_%s' % metric_key] = np.mean(metric_list)
    return mean_metrics_dict


def get_best_gans(results_dict, set_key, metric_key):
    mean_metric_dict = get_percentage_version_mean_metric_dict(results_dict, set_key, metric_key)
    percentage_keys = list(mean_metric_dict.keys())
    versions_keys = list(mean_metric_dict[percentage_keys[0]].keys())
    best_gan_dict = {}
    for percentage in percentage_keys:
        best_version = None
        best_metric = 0
        for version in versions_keys:
            metric_value = mean_metric_dict[percentage][version]['mean_%s' % metric_key]
            if metric_value > best_metric:
                best_version = version
                best_metric = metric_value
        best_gan_dict[percentage] = {'best_version': best_version, 'mean_%s' % metric_key: best_metric}
    return best_gan_dict, mean_metric_dict


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    multi_runs_dict = {}
    for run_i in range(RUNS):
        # build dict of dicts:
        result_dict_for_different_versions = {}
        for v in versions:
            result_dict_for_different_versions[v] = {}
        for v in result_dict_for_different_versions.keys():
            dict_single_version = result_dict_for_different_versions[v]
            MIN_LIM = 10
            MAX_LIM = 100
            keep_samples_list = np.round(np.logspace(np.log10(MIN_LIM), np.log10(MAX_LIM), num=6)) / 100
            for keep_sample in keep_samples_list:
                main(dict_single_version, keep_sample, v)
            print(dict_single_version)
        print(result_dict_for_different_versions)
        multi_runs_dict['run%i' % run_i] = result_dict_for_different_versions

    best_gan_dict, mean_metric_dict = get_best_gans(multi_runs_dict, SET_KEY_FOR_BEST_METRIC, BEST_METRIC_KEY)

    check_dir(os.path.join('results', FOLDER_TO_SAVE_IN))
    pickle.dump(multi_runs_dict, open(
        os.path.join('results', FOLDER_TO_SAVE_IN, str(RUNS) + '_runs' + RESULTS_NAME + '_'.join(versions) + '.pkl'),
        "wb"))
    pickle.dump(mean_metric_dict, open(
        os.path.join('results', FOLDER_TO_SAVE_IN,
                     'mean_metric_dict_' + str(RUNS) + '_runs' + RESULTS_NAME + '_'.join(versions) + '.pkl'), "wb"))
    pickle.dump(best_gan_dict, open(
        os.path.join('results', FOLDER_TO_SAVE_IN,
                     'best_gan_dict_' + str(RUNS) + '_runs' + RESULTS_NAME + '_'.join(versions) + '.pkl'), "wb"))
    print('RESULTS')
    print('multi_runs_dict\n', multi_runs_dict)
    print('mean_metric_dict\n', mean_metric_dict)
    print('best_gan_dict\n', best_gan_dict)
