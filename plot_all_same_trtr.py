import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import  matplotlib
#matplotlib.use('agg')

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ''))
sys.path.append(PATH_TO_PROJECT)

N_SIGMAS = 1


def get_all_ro_plot(results_list, metric_to_plot, set_to_plot):
    all_results_metrics_list = []
    for result in results_list:
        percentages_str = list(result.keys())
        percentages_float = [float(i) for i in result.keys()]
        metrics_list = []
        for percentage in percentages_str:
            metrics_list.append(result[percentage][set_to_plot][metric_to_plot])
        all_results_metrics_list.append(metrics_list)
    all_results_metrics_list = np.array(all_results_metrics_list)
    all_results_metrics_mean = np.mean(all_results_metrics_list, axis=0)
    all_results_metrics_std = np.std(all_results_metrics_list, axis=0)
    all_results_metrics_low_bound = all_results_metrics_mean - N_SIGMAS * all_results_metrics_std
    all_results_metrics_up_bound = np.clip(all_results_metrics_mean + N_SIGMAS * all_results_metrics_std, None, 1.0)
    return percentages_float, all_results_metrics_mean, all_results_metrics_low_bound, all_results_metrics_up_bound


def plot_metric(results_list, set_to_plot, metric_to_plot, x_axis_name, y_axis_name, fig_size=8, plot_label=''):
    if isinstance(set_to_plot, str):
        set_to_plot = [set_to_plot for i in range(len(metric_to_plot))]
    plt.figure(figsize=(fig_size, fig_size))
    for result_idx in range(len(results_list)):
        percentages_float, all_results_metrics_mean, all_results_metrics_low_bound, all_results_metrics_up_bound = get_all_ro_plot(
            results_list[result_idx], metric_to_plot[result_idx], set_to_plot[result_idx])
        print(all_results_metrics_mean)
        plt.plot(percentages_float, all_results_metrics_mean, 'o-', label=plot_label[result_idx])
        plt.fill_between(percentages_float, all_results_metrics_low_bound, all_results_metrics_up_bound, alpha=.1)
    plt.title('Starlight Accuracy as a function of unbalance (Same set, with through away)')  # % set_to_plot)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    #plt.ylim(0.9,1)
    plt.legend()
    plt.savefig('fig.png')
    plt.show()



def get_results_from_path(path):
    results_dict = np.load(path)
    results = [results_dict[dict_key] for dict_key in results_dict.keys()]
    return results

results_trtr_bn = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'fine_tune',
        'trtr_FT_val_loss_batch_norm_dp_0.5_pt_20_starlight_noisy_irregular_all_same_set_amp_balanced_larger_trainv2_v3_v4_v5_v6_v7_v8_v9.pkl')
)

results_trtr_bal = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'new_results',
        'trtr_balanced_starlight_new_bal.pkl')
)

results_trtr_naive = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'new_results',
        'trtr_naive_starlight_new_naive.pkl')
)

results_trtr_basic = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'new_results',
        'trtr_basic_starlight_new_basic.pkl')
)

results_fine_tunning = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'new_results',
        'fine_tune_lr3_dp0.5_starlight_new_bal_best_versions.pkl')
)

results_fine_tune_3 = get_results_from_path(
    os.path.join(
        PATH_TO_PROJECT, 'results', 'fine_tune',
        'single_gan_resultsfinetune_loss_stop_lr3.0_batch_norm_dp_0.5_pt_20_starlight_noisy_irregular_all_same_set_amp_balanced_larger_trainv2_v3_v4_v5_v6_v7_v8_v9.pkl')
)





set_to_plot = 'testing'
metric_to_plot_tstr = 'Test accuracy'
metric_to_plot_trtr = 'Test accuracy on real'
# 'Test accuracy'

plot_metric([results_trtr_bn, results_trtr_bal, results_trtr_basic, results_trtr_naive, results_fine_tunning, results_fine_tune_3], set_to_plot,
            [metric_to_plot_trtr, metric_to_plot_trtr, metric_to_plot_trtr, metric_to_plot_trtr, metric_to_plot_trtr, metric_to_plot_trtr],
            '% of least populated classes kept', 'Accuracy', plot_label = ['TRTR_BN', 'TRTR_bal', 'TRTR_basic', 'TRTR_naive',
                                                                           'Fine_tunning', 'Fine_tune_old'
                                                               ])

