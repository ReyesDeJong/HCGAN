import matplotlib.pyplot as plt
import numpy as np
result_dict_v1 = {'0.2': {'training': {'IS Mean': 2.993878648462156, 'IS Std': 0.0020906032708043402, 'ACC': 0.9866450617434066, 'VAL_ACC': 0.9941456790432519, 'LOSS': 0.03530485968131653, 'VAL_LOSS': 0.016469503936512156}, 'testing': {'test loss': 2.21709259668986, 'Test accuracy': 0.7922222222222223, 'auc roc': 0.9653592592592593}}, '0.28': {'training': {'IS Mean': 2.9892236221665103, 'IS Std': 0.002254077731151808, 'ACC': 0.9890148148433073, 'VAL_ACC': 0.9937540743910236, 'LOSS': 0.03005297101627874, 'VAL_LOSS': 0.018820684774105843}, 'testing': {'test loss': 3.0871464257770116, 'Test accuracy': 0.7422222222222222, 'auc roc': 0.9514157407407406}}, '0.38': {'training': {'IS Mean': 2.9877267688778653, 'IS Std': 0.003293772746256643, 'ACC': 0.9854785908161463, 'VAL_ACC': 0.991054742427247, 'LOSS': 0.04045486550694054, 'VAL_LOSS': 0.02386967666127838}, 'testing': {'test loss': 1.0710941558082898, 'Test accuracy': 0.8777777777777778, 'auc roc': 0.976255787037037}}, '0.53': {'training': {'IS Mean': 2.987294583588284, 'IS Std': 0.0026694433451213274, 'ACC': 0.9855064815033364, 'VAL_ACC': 0.990524074312272, 'LOSS': 0.03927096519713825, 'VAL_LOSS': 0.026484241665286633}, 'testing': {'test loss': 1.867503046989441, 'Test accuracy': 0.8, 'auc roc': 0.97038125}}, '0.72': {'training': {'IS Mean': 2.9943203428282326, 'IS Std': 0.0024265806834613914, 'ACC': 0.9883286738474306, 'VAL_ACC': 0.9918623657151361, 'LOSS': 0.03322886227292083, 'VAL_LOSS': 0.02542037498028742}, 'testing': {'test loss': 1.0489188285909283, 'Test accuracy': 0.8894444444444445, 'auc roc': 0.9805247685185186}}, '1.0': {'training': {'IS Mean': 2.988992038952112, 'IS Std': 0.002391307364333524, 'ACC': 0.9868283688243117, 'VAL_ACC': 0.9910751769629497, 'LOSS': 0.03682886243994705, 'VAL_LOSS': 0.028454348320765835}, 'testing': {'test loss': 0.5981010508094914, 'Test accuracy': 0.9227777777777778, 'auc roc': 0.9911828703703703}}}
result_dict_v2 = {'0.2': {'training': {'IS Mean': 2.9960315564395037, 'IS Std': 0.001420912693643171, 'ACC': 0.9926542222298513, 'VAL_ACC': 0.996624000298182, 'LOSS': 0.0220074387466938, 'VAL_LOSS': 0.01102509829728114}, 'testing': {'test loss': 0.7402012877606062, 'Test accuracy': 0.9177777777777778, 'auc roc': 0.9821972222222222}}, '0.28': {'training': {'IS Mean': 2.9941524982938477, 'IS Std': 0.00283193908148229, 'ACC': 0.9899935587944423, 'VAL_ACC': 0.9934389697418887, 'LOSS': 0.02759202868147391, 'VAL_LOSS': 0.019900333376111332}, 'testing': {'test loss': 1.6934950510660807, 'Test accuracy': 0.8338888888888889, 'auc roc': 0.9579766203703702}}, '0.38': {'training': {'IS Mean': 2.992595034859901, 'IS Std': 0.00199630251306342, 'ACC': 0.992217840389616, 'VAL_ACC': 0.9947956184897624, 'LOSS': 0.022316496387747856, 'VAL_LOSS': 0.01600007383016656}, 'testing': {'test loss': 1.3229616685708363, 'Test accuracy': 0.8533333333333334, 'auc roc': 0.9687659722222222}}, '0.53': {'training': {'IS Mean': 2.9927092331118232, 'IS Std': 0.002299277188341128, 'ACC': 0.9838645645954948, 'VAL_ACC': 0.9890306308577369, 'LOSS': 0.04579768943974228, 'VAL_LOSS': 0.03024034341958327}, 'testing': {'test loss': 0.8605877656895771, 'Test accuracy': 0.8977777777777778, 'auc roc': 0.9886530092592593}}, '0.72': {'training': {'IS Mean': 2.979522980600027, 'IS Std': 0.003912068432044959, 'ACC': 0.9795512820746481, 'VAL_ACC': 0.9839703702964675, 'LOSS': 0.05515963700107403, 'VAL_LOSS': 0.04260005430841335}, 'testing': {'test loss': 1.4462032800250644, 'Test accuracy': 0.8111111111111111, 'auc roc': 0.969287037037037}}, '1.0': {'training': {'IS Mean': 2.986116449878552, 'IS Std': 0.0030807626434000185, 'ACC': 0.9871694444701785, 'VAL_ACC': 0.9927158730983735, 'LOSS': 0.03520865276828917, 'VAL_LOSS': 0.0228705822044795}, 'testing': {'test loss': 0.7483297014898724, 'Test accuracy': 0.8994444444444445, 'auc roc': 0.9839583333333333}}}
results_list = [result_dict_v1, result_dict_v2]
#percentages_str = list(results_list.keys())
#percentages_float = [float(i) for i in results_list.keys()]
set_to_plot = 'testing'
metric_to_plot = 'Test accuracy'

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
    all_results_metrics_low_bound = all_results_metrics_mean - 2 * all_results_metrics_std
    all_results_metrics_up_bound = np.clip(all_results_metrics_mean + 2 * all_results_metrics_std, None, 1.0)

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

