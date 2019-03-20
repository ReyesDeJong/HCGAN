import matplotlib.pyplot as plt
import numpy as np
import os
import sys
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ''))
sys.path.append(PATH_TO_PROJECT)

N_SIGMAS = 1

dicts = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'trts_same_all_v2_v3_v4_v5_v6_v7_v9_v10.pkl'))
results_list = [dicts[dict_key] for dict_key in dicts.keys()]
#percentages_str = list(results_list.keys())
#percentages_float = [float(i) for i in results_list.keys()]
set_to_plot = 'testing'
metric_to_plot = 'Test accuracy on real'#'Test accuracy'

#metrics_list = []
#for percentage in percentages_str:
#    metrics_list.append(results_dict[percentage][set_to_plot][metric_to_plot])

#plt.plot(percentages_float, metrics_list)
#plt.show()

def plot_metric(results_list, set_to_plot, metric_to_plot, x_axis_name, y_axis_name, fig_size=8):
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

    print(all_results_metrics_list)
    print(all_results_metrics_mean)
    print(all_results_metrics_std)
    print(all_results_metrics_low_bound)
    print(all_results_metrics_up_bound)

    plt.figure(figsize=(fig_size, fig_size))
    plt.plot(percentages_float, all_results_metrics_mean, 'o-')
    plt.fill_between(percentages_float, all_results_metrics_low_bound, all_results_metrics_up_bound, alpha=.1)
    plt.title('TSTR as a function of unbalance (%% least populated classes kept in %s set)' % set_to_plot)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.show()

plot_metric(results_list, set_to_plot, metric_to_plot, '% of least populated classes kept', 'TSTR Accuracy')
#plot_metric(percentages_str, set_to_plot, 'auc roc', '% of least populated classes kept', 'TSTR AUC-ROC')

