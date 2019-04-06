from model_keras import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import my_callbacks
import numpy as np
from keras.utils import to_categorical
import os
import sys
from sklearn.utils import shuffle
from keras.models import load_model
import keras
import keras.backend as K

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ''))
sys.path.append(PATH_TO_PROJECT)

MIN_LIM = 10
MAX_LIM = 100
LR_VAL_MULT = 3
DROP_OUT_RATE = 0.5
PATIENCE = 30
PATIENCE_FINE = 200
TEST_SET_KEY = 'testing'
TEST_METRIC_KEY = 'Test accuracy on real'
SET_KEY_FOR_BEST_METRIC = 'training'
BEST_METRIC_KEY = 'VAL_ACC'
EARLY_STOP_ON = 'val_loss'
EARLY_STOP_ON_COD = 'min'
BN_CONDITION = 'batch_norm_'
BASE_REAL_NAME = 'starlight_new_bal_'
AUGMENTED_OR_NOT_EXTRA_STR = ''  # '_augmented_50-50'  # #
versions = ['', 'v2', 'v3', 'v4', 'v5']
RUNS = 10
RESULTS_NAME = 'fine_tune_lr%s_dp%.1f%s_%s' % (
    LR_VAL_MULT, DROP_OUT_RATE, AUGMENTED_OR_NOT_EXTRA_STR, BASE_REAL_NAME)
FOLDER_TO_SAVE_IN = 'new_results'

date = '2803'


def targets_to_numbers(targets):
    target_keys = np.unique(targets)
    target_keys_idxs = np.argsort(np.unique(targets))
    targets_as_numbers = target_keys_idxs[np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]
    return targets_as_numbers


def get_data_from_set(set, magnitude_key, time_key):
    magnitudes = np.asarray(set[magnitude_key])
    time = np.asarray(set[time_key])
    x = np.stack((magnitudes, time), axis=-1)
    x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
    y = np.asarray(set['class'])
    x, y = shuffle(x, y, random_state=42)
    y = targets_to_numbers(y)
    y = to_categorical(y)  # to one-hot
    return x, y


def read_data_irregular_sampling(file, magnitude_key='original_magnitude', time_key='time', verbose=False):
    dataset_partitions = np.load(file)
    if verbose:
        print(dataset_partitions[0].keys())
    x_train, y_train = get_data_from_set(dataset_partitions[0], magnitude_key, time_key)
    x_val, y_val = get_data_from_set(dataset_partitions[1], magnitude_key, time_key)
    x_test, y_test = get_data_from_set(dataset_partitions[2], magnitude_key, time_key)
    return x_train, y_train, x_val, y_val, x_test, y_test


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(result_dict={}, percentage_of_samples_to_keep_for_imbalance=1.0, v=''):
    real_data_folder = os.path.join('datasets_original', 'REAL')
    dataset_real_pkl = '%s%.2f.pkl' % (BASE_REAL_NAME, percentage_of_samples_to_keep_for_imbalance)
    syn_data_name = os.path.join('%s%s%.2f' % (BASE_REAL_NAME, v, percentage_of_samples_to_keep_for_imbalance))

    percentage_of_samples_to_keep_for_imbalance_key = str(percentage_of_samples_to_keep_for_imbalance)
    result_dict[percentage_of_samples_to_keep_for_imbalance_key] = {'training': {}, 'testing': {}}
    print("\nREAL Training set to load %s" % dataset_real_pkl)
    print("SYN Training set to load %s" % syn_data_name)

    dataset_syn_pkl = syn_data_name + '_generated.pkl'

    # load syn and real data
    x_train_syn, y_train_syn, x_val_syn, y_val_syn, x_test_syn, y_test_syn = read_data_irregular_sampling(
        os.path.join('TSTR_data', 'generated', syn_data_name, dataset_syn_pkl), magnitude_key='generated_magnitude',
        time_key='time')
    x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = read_data_irregular_sampling(
        os.path.join('TSTR_data', real_data_folder, dataset_real_pkl), magnitude_key='generated_magnitude',
        time_key='time')

    ## Train on synthetic
    print('\nTraining new model\n')
    batch_size = 512
    epochs = 10000
    num_classes = 3
    # choose model
    m = Model_(batch_size, 100, num_classes, drop_rate=DROP_OUT_RATE)
    if BN_CONDITION == 'batch_norm_':
        model = m.cnn2_batch()
    else:
        model = m.cnn2()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ## callbacks
    history = my_callbacks.Histories()
    weight_folder = os.path.join('TSTR_' + date, 'train', RESULTS_NAME, syn_data_name)
    check_dir(weight_folder)
    checkpoint = ModelCheckpoint(os.path.join(weight_folder, 'weights.best.trainonsynthetic.hdf5'),
                                 monitor=EARLY_STOP_ON, verbose=1, save_best_only=True, mode=EARLY_STOP_ON_COD)
    earlyStopping = EarlyStopping(monitor=EARLY_STOP_ON, min_delta=0.00000001, patience=PATIENCE, verbose=1,
                                  mode=EARLY_STOP_ON_COD)

    model.fit(x_train_syn, y_train_syn, epochs=epochs, batch_size=batch_size, validation_data=(x_val_real, y_val_real),
              callbacks=[history,
                         checkpoint,
                         earlyStopping
                         ])
    model = load_model(os.path.join(weight_folder, 'weights.best.trainonsynthetic.hdf5'))

    print('Syn Training metrics:')
    score_train = model.evaluate(x_train_syn, y_train_syn, verbose=1)
    score_val = model.evaluate(x_val_real, y_val_real, verbose=1)
    score_tstr = model.evaluate(x_test_real, y_test_real, verbose=1)
    print('ACC : ', score_train[1])
    print('VAL_ACC : ', score_val[1])
    print('LOSS : ', score_train[0])
    print('VAL_LOSS : ', score_val[0])
    print('TSTR loss: %f ;-; accuracy: %f' % (score_tstr[0], score_tstr[1]))
    result_dict[percentage_of_samples_to_keep_for_imbalance_key]['testing'] = {
        'tstr loss': score_tstr[0], 'tstr accuracy': score_tstr[1]
    }

    # fine tunning
    K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * LR_VAL_MULT)
    checkpoint = ModelCheckpoint(os.path.join(weight_folder, 'weights.best.trainfinetune.hdf5'),
                                 monitor=EARLY_STOP_ON, verbose=1, save_best_only=True, mode=EARLY_STOP_ON_COD)
    earlyStopping = EarlyStopping(monitor=EARLY_STOP_ON, min_delta=0.00000001, patience=PATIENCE_FINE, verbose=1,
                                  mode=EARLY_STOP_ON_COD)

    model.fit(x_train_real, y_train_real, epochs=epochs, batch_size=batch_size,
              validation_data=(x_val_real, y_val_real),
              callbacks=[history,
                         checkpoint,
                         earlyStopping
                         ])
    model = load_model(os.path.join(weight_folder, 'weights.best.trainfinetune.hdf5'))

    ## Test on real
    score_val = model.evaluate(x_val_real, y_val_real, verbose=1)
    print('fine tune VAL_ACC : ', score_val[1])
    print('fine tune VAL_LOSS : ', score_val[0])

    print('\nTest metrics:')
    print('\nTest on real:')
    score = model.evaluate(x_test_real, y_test_real, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result_dict[percentage_of_samples_to_keep_for_imbalance_key]['testing']['test loss on real'] = score[0]
    result_dict[percentage_of_samples_to_keep_for_imbalance_key]['testing']['Test accuracy on real'] = score[1]
    result_dict[percentage_of_samples_to_keep_for_imbalance_key]['training'] = {
        'VAL_ACC': score_val[1], 'TRAIN_ACC': score_train[1],
        'TRAIN_LOSS': score_train[0], 'VAL_LOSS': score_val[0]
    }

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


# create dict to plot format; runs-percentages
def from_best_gan_get_metric(results_dict, best_gans_dict, set_key, metric_key):
    mean_metrics_dict = get_percentage_version_mean_metric_dict(results_dict, set_key, metric_key)
    metrics_to_return_dict = {}
    # create empty dict
    runs_keys = list(results_dict.keys())
    versions_keys = list(results_dict[runs_keys[0]].keys())
    percentage_keys = list(results_dict[runs_keys[0]][versions_keys[0]].keys())
    for run in runs_keys:
        metrics_to_return_dict[run] = {}
        for percentage in percentage_keys:
            metrics_to_return_dict[run][percentage] = {}
            metrics_to_return_dict[run][percentage][set_key] = {}
            metrics_to_return_dict[run][percentage][set_key][metric_key] = {}
    # fill new dict
    print(metrics_to_return_dict)
    for percentage in percentage_keys:
        best_gan_version = best_gans_dict[percentage]['best_version']
        runs_of_best_gan = mean_metrics_dict[percentage][best_gan_version]
        print('best_gan_version : %s VAL mean %f Test mean %f' % (best_gan_version,
                                                                  best_gans_dict[percentage]['mean_%s' % BEST_METRIC_KEY],
                                                                  runs_of_best_gan['mean_%s' % TEST_METRIC_KEY],))
        for run in runs_keys:
            metrics_to_return_dict[run][percentage][set_key][metric_key] = runs_of_best_gan[run]
    return metrics_to_return_dict


def print_gans_means(results_dict, set_key, metric_key):
    mean_metrics_dict = get_percentage_version_mean_metric_dict(results_dict, set_key, metric_key)
    runs_keys = list(results_dict.keys())
    versions_keys = list(results_dict[runs_keys[0]].keys())
    percentage_keys = list(results_dict[runs_keys[0]][versions_keys[0]].keys())
    for percentage in percentage_keys:
        str_to_print = '\nMean %s, for gan versions trained with keep percentage %s:\n' % (metric_key, percentage)
        for version in versions_keys:
            metric_value = mean_metrics_dict[percentage][version]['mean_%s' % metric_key]
            str_to_print += '%s:%.4f; ' % (version, metric_value)
        print(str_to_print)


if __name__ == '__main__':
    multi_runs_dict = {}
    # different runs for every version and percentage
    for run_i in range(RUNS):
        # build dict of dicts:
        result_dict_for_different_versions = {}
        for v in versions:
            result_dict_for_different_versions[v] = {}
        # get percentage results for different version
        for v in result_dict_for_different_versions.keys():
            dict_single_version = result_dict_for_different_versions[v]
            keep_samples_list = np.round(np.logspace(np.log10(MIN_LIM), np.log10(MAX_LIM), num=6)) / 100
            for keep_sample in keep_samples_list:
                print('\n\nRUN %i, keep %.2f%%' % (run_i, keep_sample))
                main(dict_single_version, keep_sample, v)
        multi_runs_dict['run%i' % run_i] = result_dict_for_different_versions

    best_gan_dict, _ = get_best_gans(multi_runs_dict, SET_KEY_FOR_BEST_METRIC, BEST_METRIC_KEY)
    best_versions_test_results_dict = from_best_gan_get_metric(multi_runs_dict, best_gan_dict, TEST_SET_KEY, TEST_METRIC_KEY)

    results_path = os.path.join(PATH_TO_PROJECT, 'results', FOLDER_TO_SAVE_IN)
    check_dir(results_path)
    pickle.dump(multi_runs_dict, open(
        os.path.join(results_path, str(RUNS) + '_runs' + RESULTS_NAME + ''.join(versions) + '.pkl'), "wb"))
    pickle.dump(best_versions_test_results_dict, open(
        os.path.join(results_path, RESULTS_NAME + 'best_versions' + '.pkl'), "wb"))
    print('RESULTS')
    print('best_gan_dict\n', best_gan_dict)
    print_gans_means(multi_runs_dict, SET_KEY_FOR_BEST_METRIC, BEST_METRIC_KEY)