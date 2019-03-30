import matplotlib.pyplot as plt
import numpy as np
import os
import sys
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ''))
sys.path.append(PATH_TO_PROJECT)

N_SIGMAS = 1

def get_all_ro_plot(results_list, metric_to_plot):
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
    plt.figure(figsize=(fig_size, fig_size))
    for result_idx in range(len(results_list)):
        percentages_float, all_results_metrics_mean, all_results_metrics_low_bound, all_results_metrics_up_bound = get_all_ro_plot(results_list[result_idx], metric_to_plot[result_idx])
        plt.plot(percentages_float, all_results_metrics_mean, 'o-', label=plot_label[result_idx])
        plt.fill_between(percentages_float, all_results_metrics_low_bound, all_results_metrics_up_bound, alpha=.1)
    plt.title('Starlight Accuracy as a function of unbalance (Same set, with through away)')# % set_to_plot)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    #plt.ylim(0.5,1)
    plt.legend()
    plt.show()

#same set tstr
"""
result_dict_v1 = {'0.2': {'training': {'IS Mean': 2.9966588391588695, 'IS Std': 0.0015259399385621725, 'ACC': 0.9930140255175869, 'VAL_ACC': 0.9964852459607012, 'LOSS': 0.019392131881038457, 'VAL_LOSS': 0.011938008783716146}, 'testing': {'test loss': 1.7816211772694643, 'Test accuracy': 0.8372222222222222, 'auc roc': 0.9726009259259261}}, '0.28': {'training': {'IS Mean': 2.99102316833999, 'IS Std': 0.0014182938649434412, 'ACC': 0.9861706093371128, 'VAL_ACC': 0.9958652329995212, 'LOSS': 0.03872168796967087, 'VAL_LOSS': 0.012991281581482135}, 'testing': {'test loss': 1.332877313189674, 'Test accuracy': 0.8677777777777778, 'auc roc': 0.9798888888888889}}, '0.38': {'training': {'IS Mean': 2.9936819586290526, 'IS Std': 0.0026536348976116367, 'ACC': 0.9863351852072609, 'VAL_ACC': 0.9933728395179463, 'LOSS': 0.036686009859301746, 'VAL_LOSS': 0.018536861792325984}, 'testing': {'test loss': 1.0294477348609103, 'Test accuracy': 0.8688888888888889, 'auc roc': 0.9871407407407409}}, '0.53': {'training': {'IS Mean': 2.9753359474009935, 'IS Std': 0.004066358748233832, 'ACC': 0.9821563492335971, 'VAL_ACC': 0.991047619297391, 'LOSS': 0.04925517260056167, 'VAL_LOSS': 0.025296551671194}, 'testing': {'test loss': 0.8984635574618994, 'Test accuracy': 0.8661111111111112, 'auc roc': 0.989388888888889}}, '0.72': {'training': {'IS Mean': 2.9804478881013354, 'IS Std': 0.004322770381102516, 'ACC': 0.985521863831756, 'VAL_ACC': 0.9924759857328134, 'LOSS': 0.04175228811265454, 'VAL_LOSS': 0.021908857124635958}, 'testing': {'test loss': 0.9428277542426561, 'Test accuracy': 0.8622222222222222, 'auc roc': 0.9903300925925925}}, '1.0': {'training': {'IS Mean': 2.9892725048936453, 'IS Std': 0.0021565941120760402, 'ACC': 0.9889215167755383, 'VAL_ACC': 0.9937707232485373, 'LOSS': 0.03383600787899178, 'VAL_LOSS': 0.02147953961582315}, 'testing': {'test loss': 0.4196034456564425, 'Test accuracy': 0.9372222222222222, 'auc roc': 0.9950393518518519}}}
result_dict_v2 = {'0.2': {'training': {'IS Mean': 2.9964953659769398, 'IS Std': 0.002105531130922382, 'ACC': 0.9877468013606088, 'VAL_ACC': 0.9953319866138678, 'LOSS': 0.0340243090115265, 'VAL_LOSS': 0.01673831675839183}, 'testing': {'test loss': 2.127298148122758, 'Test accuracy': 0.8166666666666667, 'auc roc': 0.9695148148148148}}, '0.28': {'training': {'IS Mean': 2.993123684488122, 'IS Std': 0.002978307925718642, 'ACC': 0.990063565911187, 'VAL_ACC': 0.9971617571246715, 'LOSS': 0.030184552700538905, 'VAL_LOSS': 0.008273107149789276}, 'testing': {'test loss': 1.3533534638397395, 'Test accuracy': 0.87, 'auc roc': 0.977124074074074}}, '0.38': {'training': {'IS Mean': 2.995091675059462, 'IS Std': 0.0024379333511039027, 'ACC': 0.9865603486110962, 'VAL_ACC': 0.9949960785907339, 'LOSS': 0.03404197589110569, 'VAL_LOSS': 0.01560135443550125}, 'testing': {'test loss': 0.8522711915684017, 'Test accuracy': 0.905, 'auc roc': 0.9868083333333333}}, '0.53': {'training': {'IS Mean': 2.9906212884059458, 'IS Std': 0.004025698775941561, 'ACC': 0.9902779956589078, 'VAL_ACC': 0.995344662386036, 'LOSS': 0.02764448864576508, 'VAL_LOSS': 0.013319632606878413}, 'testing': {'test loss': 0.743193767352754, 'Test accuracy': 0.9105555555555556, 'auc roc': 0.9869490740740741}}, '0.72': {'training': {'IS Mean': 2.9862168060193803, 'IS Std': 0.00485598809144394, 'ACC': 0.9822523297836275, 'VAL_ACC': 0.9895713261149692, 'LOSS': 0.04885116108194607, 'VAL_LOSS': 0.02935231226921135}, 'testing': {'test loss': 0.8841740762909952, 'Test accuracy': 0.8838888888888888, 'auc roc': 0.9920719907407407}}, '1.0': {'training': {'IS Mean': 2.9911427943180713, 'IS Std': 0.00265027562476238, 'ACC': 0.9890888889051253, 'VAL_ACC': 0.9917462367467982, 'LOSS': 0.03137594700976221, 'VAL_LOSS': 0.025034144106839046}, 'testing': {'test loss': 0.5793040703613467, 'Test accuracy': 0.9233333333333333, 'auc roc': 0.9896881944444443}}}
some_dicts = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'v2_v3_v4_v5_v6_v10.pkl'))
#result_dict_v2_2 = some_dicts['v2']
result_dict_v3 = some_dicts['v3']
result_dict_v4 = some_dicts['v4']
result_dict_v5 = some_dicts['v5']
result_dict_v6 = some_dicts['v6']
some_other_dicts = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'v7_v9.pkl'))
result_dict_v7 = some_other_dicts['v7']
result_dict_v9 = some_other_dicts['v9']
result_dict_v10 = some_dicts['v10']
results_tstr = [result_dict_v1, result_dict_v2, result_dict_v3, result_dict_v4, result_dict_v5, result_dict_v6, result_dict_v7, result_dict_v9, result_dict_v10]

"""
#dicts_tstr = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'tstr_same_all_v2_v3_v4_v5_v6_v7_v8_v9_v10.pkl'))
#results_tstr = [dicts_tstr[dict_key] for dict_key in dicts_tstr.keys()]

dicts_trtr = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'trts_dp_0.5_pt_20__starlight_noisy_irregular_all_same_set_amp_balanced_larger_trainv2_v3_v4_v5_v6_v7_v8_v9.pkl'))
results_trtr = [dicts_trtr[dict_key] for dict_key in dicts_trtr.keys()]

#undo this, just to compare old plot befora cota error
dicts_trtr_aug = np.load(os.path.join(PATH_TO_PROJECT, 'results_old', 'same_set', 'trts_starlight_noisy_irregular_all_same_set_amp_balanced_larger_trainv2_v3_v4_v5_v6_v7_v8_v9.pkl'))
#dicts_trtr_aug = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'trts__augmented_amp_balanced_larger_trainv2_v3_v4_v5.pkl'))
results_trtr_aug = [dicts_trtr_aug[dict_key] for dict_key in dicts_trtr_aug.keys()]

#dicts_trtr_aug_50_50 = np.load(os.path.join(PATH_TO_PROJECT, 'results', 'same_set', 'trts__augmented_amp_balanced_larger_train_50-50v2_v3_v4_v5.pkl'))
#results_trtr_aug_50_50 = [dicts_trtr_aug_50_50[dict_key] for dict_key in dicts_trtr_aug_50_50.keys()]

set_to_plot = 'testing'
metric_to_plot_tstr = 'Test accuracy'
metric_to_plot_trtr = 'Test accuracy on real'#'Test accuracy'

#plot_metric([results_trtr, results_trtr_aug, results_trtr_aug_50_50], set_to_plot, [metric_to_plot_trtr, metric_to_plot_trtr, metric_to_plot_trtr],
#            '% of least populated classes kept', 'Accuracy', plot_label=['TRTR', 'Augmented', 'Augmented_50-50'])

plot_metric([results_trtr, results_trtr_aug], set_to_plot, [metric_to_plot_trtr, metric_to_plot_trtr],
            '% of least populated classes kept', 'Accuracy', plot_label=['TRTR', 'TRT_old'])


#plot_metric([results_tstr, results_trtr, results_trtr_aug], set_to_plot, [metric_to_plot_tstr, metric_to_plot_trtr, metric_to_plot_trtr],
#            '% of least populated classes kept', 'Accuracy', plot_label=['TSTR', 'TRTR', 'Augmented'])

#plot_metric([results_tstr, results_trtr], set_to_plot, [metric_to_plot_tstr, metric_to_plot_trtr],
#            '% of least populated classes kept', 'Accuracy', plot_label=['TSTR', 'TRTR'])
