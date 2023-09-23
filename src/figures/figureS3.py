import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from common import average_roc, average_pr, get_y_ture_y_pred_y_score


def get_mean_and_std(y_ture_list, y_pred_list, y_score_list):
    all_evaluation_metrics = {}
    for idx in range(len(y_ture_list)):
        y_ture = y_ture_list[idx]
        y_pred = y_pred_list[idx]
        y_score = y_score_list[idx]
        _, res = evaluate(y_ture, y_pred, y_score)
        for key in res:
            all_evaluation_metrics.setdefault(key, []).append(res[key])
    results = {}
    for key, value in zip(all_evaluation_metrics.keys(), all_evaluation_metrics.values()):
        mean_value = np.mean(value)
        std_value = np.std(value)
        mean_value = round(float(mean_value), 3)
        std_value = round(float(std_value), 3)
        results[key] = f'{mean_value}+-{std_value}'
    return results, all_evaluation_metrics


def evaluate(y_true, y_pred, y_score):
    """
    compute the 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC', 'F1-score',
    'AUC-ROC', 'AUC-PR'.
    :param y_pred:
    :param y_score: the outputs of model
    :param y_true: the real labels of samples
    :return: confusion_matrix, evaluate_results
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index in range(len(y_pred)):
        if y_pred[index] == 1 and y_true[index] == 1:
            tp += 1
        elif y_pred[index] == 1 and y_true[index] == 0:
            fp += 1
        elif y_pred[index] == 0 and y_true[index] == 1:
            fn += 1
        else:
            tn += 1
    confusion_matrix = {'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp}
    if (tp + tn + fp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = 0
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if (tp + fn) != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0
    if (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5) != 0:
        mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    else:
        mcc = 0
    if (fp + tn) != 0:
        specificity = tn / (fp + tn)
    else:
        specificity = 0
    result = {'Accuracy': accuracy, 'Precision': precision, 'Sensitivity': sensitivity, 'Specificity': specificity,
              'MCC': mcc, 'F1-score': f1_score(y_true, y_pred), 'AUC-ROC': roc_auc_score(y_true, y_score),
              'AUC-PR': average_precision_score(y_true, y_score)}
    return confusion_matrix, result


def figure2(values, values2, values3, figurename='Figure2'):
    """
    :param values2: {'SilenceREIN': {'': [], '': []},
                    }
    :param values: {'SilenceREIN': {'Acc': , 'Sen', , 'Spe': },
                    ...}
    :return: None
    """
    colors = ['#4F9F3C', '#EF8D8E', '#599AAD', '#C799C7',
              '#f6b654', '#c7dbd5', '#4ea59f', 'teal', 'royalblue', 'darkred']
    plt.figure(figsize=(15, 15))

    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    # 柱状图
    plt.subplot(211)
    plt.text(-0.80, 1.15, "(A)", fontsize=15, fontweight='bold')
    means = {}
    stds = {}
    for model in values:
        for key in ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-score']:
            mean, std = values[model][key].split('+-')
            means.setdefault(model, []).append(float(mean))
            stds.setdefault(model, []).append(float(std))

    x_names = ['Sen', 'Spe', 'PPV', 'ACC', 'MCC', 'F1']
    x = np.arange(len(x_names))  # the label locations
    # tmp = []
    # for idx, val in enumerate(x):
    #     tmp.append(val + 0.5 * idx)
    # x = np.array(tmp)

    width = 0.1  # the width of the bars

    plt.bar(x - 0.3, means['SilenceREIN'],
            color=colors[0], width=width, label='SilenceREIN')
    plt.bar(x - 0.15, means['SilenceREIN-alt'],
            color=colors[1], width=width, label='SilenceREIN-alt')
    plt.bar(x - 0.0, means['CNN'],
            color=colors[2], width=width, label='CNN')
    plt.bar(x + 0.15, means['DeepSilencer'],
            color=colors[3], width=width,
            label='DeepSilencer')
    plt.bar(x + 0.3, means['gkmSVM'],
            color=colors[4], width=width, label='gkmSVM')

    # for idx, key in enumerate(['SilenceREIN', 'CNN', 'DeepSilencer', 'gkmSVM']):
    # for idx, key in enumerate(['SilenceREIN']):
    bias = [-0.3, -0.15, 0, 0.15, 0.3]
    for j, mer in enumerate(['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-score']):
        for idx, model in enumerate(['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']):
            y = values3[model][mer]
            # x0 = np.random.normal(bias[idx], 0.02, size=len(y))
            x0 = [bias[idx] + j] * len(y)
            plt.plot(x0, y, '.', color='r', alpha=0.8)

    # Q1, S3
    for j, mer in enumerate(['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-score']):
        for k, model in enumerate(['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']):
            y = values3[model][mer]
            y = sorted(y)
            q1 = y[1]
            q2 = y[3]
            x0 = bias[k] + j
            x = [x0, x0]
            y = [q1, q2]
            plt.plot(x, y, '-', linewidth=1, color='black')

            x = [x0 - 0.04, x0 + 0.04]
            y = [q1, q1]
            plt.plot(x, y, '-', linewidth=1, color='black')

            x = [x0 - 0.04, x0 + 0.04]
            y = [q2, q2]
            plt.plot(x, y, '-', linewidth=1, color='black')

    plt.ylim([0, 1.099])
    plt.yticks(fontname=font, weight='bold')
    plt.xticks(range(0, len(x_names)), x_names, fontname=font, fontweight='bold')

    ax = plt.gca()
    fontsize = 8

    for x1, y1 in enumerate(means['SilenceREIN']):
        std = stds['SilenceREIN'][x1]
        a = ('%.3f' % (round(y1, 3)))
        b = ('%.3f' % (round(std, 3)))
        if x1 == 0:
            y1 += 0.005
        if x1 == 1:
            y1 += 0.032
        if x1 == 2:
            y1 += 0.02
        x1 -= 0.02
        plt.text(x1 - 0.3, y1 + 0.010, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['SilenceREIN-alt']):
        std = stds['SilenceREIN-alt'][x1]
        a = ('%.3f' % (round(y1, 3)))
        b = ('%.3f' % (round(std, 3)))
        if x1 == 0:
            y1 += 0.05
            x1 -= 0.02
        if x1 == 1:
            y1 += 0.02
        if x1 == 2:
            y1 += 0.02
        x1 += 0.01
        plt.text(x1 - 0.15, y1 + 0.02, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['CNN']):
        std = stds['CNN'][x1]
        a = ('%.3f' % (round(y1, 3)))
        b = ('%.3f' % (round(std, 3)))
        if x1 == 5:
            x1 += 0.032
        plt.text(x1 - 0.0, y1 + 0.005, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['DeepSilencer']):
        std = stds['DeepSilencer'][x1]
        a = ('%.3f' % (round(y1, 3)))
        b = ('%.3f' % (round(std, 3)))
        if x1 == 0:
            x1 += 0.032
        if x1 == 3:
            y1 += 0.01
        if x1 == 5:
            x1 += 0.032
        if x1 in [1, 2, 4]:
            x1 -= 0.01
        plt.text(x1 + 0.15, y1 + 0.009, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['gkmSVM']):
        std = stds['gkmSVM'][x1]
        a = ('%.3f' % (round(y1, 3)))
        b = ('%.3f' % (round(std, 3)))
        if x1 == 4:
            y1 += 0.01
        x1 += 0.032
        plt.text(x1 + 0.3, y1 + 0.005, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')

    ax.legend(loc='upper right', prop={'family': font,
                                       'weight': 'bold'})

    # ROC
    plt.subplot(223)
    plt.text(-0.15, 1.10, "(B)", fontsize=15, fontweight='bold')
    fpr_list = {}
    tpr_list = {}
    mean_aucs = {}
    stds = {}
    for model in ['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        fpr, tpr, mean_auc, std = average_roc(y_true_list, y_score_list)
        fpr_list[model] = fpr
        tpr_list[model] = tpr
        mean_aucs[model] = mean_auc
        stds[model] = std
    for i, model in enumerate(['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']):
        print(f'ROC:-{model}')
        fpr = fpr_list[model]
        tpr = tpr_list[model]
        a = ('%.3f' % (round(mean_aucs[model], 3)))
        b = ('%.3f' % (round(stds[model], 3)))
        plt.plot(fpr, tpr, lw=3, color=colors[i],
                 label=f'{model} (AUROC = {a}$\pm${b})')
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', font=font, fontweight='bold')
    plt.ylabel('True Positive Rate', font=font, fontweight='bold')
    # plt.title('Receiver operating characteristic')
    plt.title('ROC', font=font, fontweight='bold')
    plt.legend(loc="lower right", prop={'family': font,
                                        'weight': 'bold'})
    plt.xticks(font=font, fontweight='bold')
    plt.yticks(font=font, fontweight='bold')

    # PR
    plt.subplot(224)
    plt.text(-0.15, 1.10, "(C)", fontsize=15, fontweight='bold')
    precision_list = {}
    recall_list = {}
    mean_auc_list = {}
    std_list = {}
    for model in ['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        precision, recall, mean_auc, std = average_pr(y_true_list, y_score_list)
        precision_list[model] = (precision)
        recall_list[model] = (recall)
        mean_auc_list[model] = (mean_auc)
        std_list[model] = (std)
    for i, model in enumerate(['SilenceREIN', 'SilenceREIN-alt', 'CNN', 'DeepSilencer', 'gkmSVM']):
        print(f'PR:-{model}')
        recall = recall_list[model]
        precision = precision_list[model]
        a = ('%.3f' % (round(mean_auc_list[model], 3)))
        b = ('%.3f' % (round(std_list[model], 3)))
        plt.plot(recall, precision, lw=2, color=colors[i],
                 label=f'{model} (AUPR = {a}$\pm${b})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', font=font, fontweight='bold')
    plt.ylabel('Precision', font=font, fontweight='bold')
    # plt.title('Precision Recall Curve')
    plt.title('PR', font=font, fontweight='bold')
    plt.legend(loc="lower right", prop={'family': font,
                                        'weight': 'bold'})
    plt.xticks(font=font, fontweight='bold')
    plt.yticks(font=font, fontweight='bold')
    plt.savefig(f'{figurename}.svg', format='SVG', dpi=1200)
    plt.savefig(f'{figurename}.png', format='PNG', dpi=300)
    plt.show()


if __name__ == '__main__':
    # K562 HiChIP------------------------------------------------------------------
    value2 = {}
    filenames_SilenceREIN = ['data/FigureS3/SilenceREIN-K562-HiChIP-0.json',
                             'data/FigureS3/SilenceREIN-K562-HiChIP-1.json',
                             'data/FigureS3/SilenceREIN-K562-HiChIP-2.json',
                             'data/FigureS3/SilenceREIN-K562-HiChIP-3.json',
                             'data/FigureS3/SilenceREIN-K562-HiChIP-4.json',
                             ]
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_SilenceREIN)
    value2['SilenceREIN'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_SilenceREIN, values_list_SilenceREIN = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_SilenceREIN)

    filenames_SilenceREIN_alt = ['data/FigureS3/SilenceREIN-alt-K562-HiChIP-0.json',
                                 'data/FigureS3/SilenceREIN-alt-K562-HiChIP-1.json',
                                 'data/FigureS3/SilenceREIN-alt-K562-HiChIP-2.json',
                                 'data/FigureS3/SilenceREIN-alt-K562-HiChIP-3.json',
                                 'data/FigureS3/SilenceREIN-alt-K562-HiChIP-4.json',
                                 ]
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_SilenceREIN_alt)
    value2['SilenceREIN-alt'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_SilenceREIN_alt, values_list_SilenceREIN_alt = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_SilenceREIN_alt)

    filenames_CNN = []
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_CNN)
    value2['CNN'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_CNN, values_list_CNN = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_CNN)

    filenames_DeepSilencer = []
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_DeepSilencer)
    value2['DeepSilencer'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_DeepSilencer, _values_list_DeepSilencer = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_DeepSilencer)

    t = os.getcwd()
    filenames_gkmSVM = []
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_gkmSVM)
    value2['gkmSVM'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_gkmSVM, values_list_gkmSVM = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_gkmSVM)

    values = {'SilenceREIN': values_SilenceREIN,
              'SilenceREIN-alt': values_SilenceREIN_alt,
              'DeepSilencer': values_DeepSilencer,
              'gkmSVM': values_gkmSVM,
              'CNN': values_CNN, }

    value3 = {'SilenceREIN': values_list_SilenceREIN,
              'SilenceREIN-alt': values_list_SilenceREIN_alt,
              'DeepSilencer': _values_list_DeepSilencer,
              'gkmSVM': values_list_gkmSVM,
              'CNN': values_list_CNN, }

    figure2(values, value2, value3, figurename='FigureS3')
