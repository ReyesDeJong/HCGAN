import matplotlib.pyplot as plt
import numpy as np

results_dict = {'0.2': {'training': {'IS Mean': 2.993878648462156, 'IS Std': 0.0020906032708043402, 'ACC': 0.9866450617434066, 'VAL_ACC': 0.9941456790432519, 'LOSS': 0.03530485968131653, 'VAL_LOSS': 0.016469503936512156}, 'testing': {'test loss': 2.21709259668986, 'Test accuracy': 0.7922222222222223, 'auc roc': 0.9653592592592593}}, '0.28': {'training': {'IS Mean': 2.9892236221665103, 'IS Std': 0.002254077731151808, 'ACC': 0.9890148148433073, 'VAL_ACC': 0.9937540743910236, 'LOSS': 0.03005297101627874, 'VAL_LOSS': 0.018820684774105843}, 'testing': {'test loss': 3.0871464257770116, 'Test accuracy': 0.7422222222222222, 'auc roc': 0.9514157407407406}}, '0.38': {'training': {'IS Mean': 2.9877267688778653, 'IS Std': 0.003293772746256643, 'ACC': 0.9854785908161463, 'VAL_ACC': 0.991054742427247, 'LOSS': 0.04045486550694054, 'VAL_LOSS': 0.02386967666127838}, 'testing': {'test loss': 1.0710941558082898, 'Test accuracy': 0.8777777777777778, 'auc roc': 0.976255787037037}}, '0.53': {'training': {'IS Mean': 2.987294583588284, 'IS Std': 0.0026694433451213274, 'ACC': 0.9855064815033364, 'VAL_ACC': 0.990524074312272, 'LOSS': 0.03927096519713825, 'VAL_LOSS': 0.026484241665286633}, 'testing': {'test loss': 1.867503046989441, 'Test accuracy': 0.8, 'auc roc': 0.97038125}}, '0.72': {'training': {'IS Mean': 2.9943203428282326, 'IS Std': 0.0024265806834613914, 'ACC': 0.9883286738474306, 'VAL_ACC': 0.9918623657151361, 'LOSS': 0.03322886227292083, 'VAL_LOSS': 0.02542037498028742}, 'testing': {'test loss': 1.0489188285909283, 'Test accuracy': 0.8894444444444445, 'auc roc': 0.9805247685185186}}, '1.0': {'training': {'IS Mean': 2.988992038952112, 'IS Std': 0.002391307364333524, 'ACC': 0.9868283688243117, 'VAL_ACC': 0.9910751769629497, 'LOSS': 0.03682886243994705, 'VAL_LOSS': 0.028454348320765835}, 'testing': {'test loss': 0.5981010508094914, 'Test accuracy': 0.9227777777777778, 'auc roc': 0.9911828703703703}}}
percentages_str = list(results_dict.keys())
percentages_float = [float(i) for i in results_dict.keys()]
set_to_plot = 'testing'
metric_to_plot = 'Test accuracy'

#metrics_list = []
#for percentage in percentages_str:
#    metrics_list.append(results_dict[percentage][set_to_plot][metric_to_plot])

#plt.plot(percentages_float, metrics_list)
#plt.show()

def plot_metric(percentages_str, set_to_plot, metric_to_plot, x_axis_name, y_axis_name, fig_size=8):
    percentages_float = [float(i) for i in results_dict.keys()]
    metrics_list = []
    for percentage in percentages_str:
        metrics_list.append(results_dict[percentage][set_to_plot][metric_to_plot])

    plt.figure(figsize=(fig_size, fig_size))
    plt.plot(percentages_float, metrics_list, 'o-')
    plt.title('TSTR as a function of unbalance (%% least populated classes kept in %s set)' % set_to_plot)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.show()

plot_metric(percentages_str, set_to_plot, metric_to_plot, '% of least populated classes kept', 'TSTR Accuracy')
plot_metric(percentages_str, set_to_plot, 'auc roc', '% of least populated classes kept', 'TSTR AUC-ROC')

