import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from common import average_roc, average_pr, get_y_ture_y_pred_y_score

def get_y_ture_y_pred_y_score_of_annotation(filenames, threshold=0.5):
    y_ture_list = []
    y_score_list = []
    y_pred_list = []
    for filename in filenames:
        y_true = []
        y_score = []
        y_pred = []
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split()
                label = float(values[1])
                y_true.append(label)
                y_score.append(float(values[2]))
            f.close()
        y_pred = [1 if score > threshold else 0 for score in y_score]
        y_ture_list.append(y_true)
        y_score_list.append(y_score)
        y_pred_list.append(y_pred)
    return y_ture_list, y_pred_list, y_score_list



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


def figure3(values, values2, values3):
    colors = ['#4F9F3C', '#EF8D8E', '#599AAD', '#C799C7',
              '#ee6a5b', '#f6b654', '#c7dbd5', '#4ea59f', 'teal', 'royalblue', 'darkred']
    plt.figure(figsize=(15, 15))

    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    # bar plot
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
    width = 0.15  # the width of the bars


    plt.bar(x - 0.3, means['SilenceREIN'],
            color=colors[0], width=width, label='SilenceREIN')
    plt.bar(x - 0.1, means['ChromHMM'],
            color=colors[1], width=width, label='ChromHMM')
    plt.bar(x + 0.1, means['Segway'],
            color=colors[2], width=width,
            label='Segway')
    plt.bar(x + 0.3, means["Libbrecht's Model"],
            color=colors[3], width=width, label="Libbrecht's Model")

    # random.seed(0)
    # # for idx, key in enumerate(['SilenceREIN', 'CNN', 'DeepSilencer', 'gkmSVM']):
    # # for idx, key in enumerate(['SilenceREIN']):
    bias = [-0.3, -0.1, 0.1, 0.3]
    for j, mer in enumerate(['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-score']):
        for idx, model in enumerate(['SilenceREIN']):
            y = values3[model][mer]
            # x0 = np.random.normal(bias[idx], 0.02, size=len(y))
            x0 = [bias[idx] + j] * len(y)
            plt.plot(x0, y, '.', color='r')

    # Q1, Q3
    for j, mer in enumerate(['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-score']):
        for k, model in enumerate(['SilenceREIN']):
            y = values3[model][mer]
            y = sorted(y)
            q1 = y[1]
            q2 = y[3]
            x0 = bias[k] + j
            x = [x0, x0]
            y = [q1, q2]
            plt.plot(x, y, '-', linewidth=1, color='black')

            x = [x0-0.04, x0+0.04]
            y = [q1, q1]
            plt.plot(x, y, '-', linewidth=1, color='black')

            x = [x0-0.04, x0+0.04]
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
        plt.text(x1 - 0.3, y1 + 0.008, f"{a}±{b}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['ChromHMM']):
        std = stds['ChromHMM'][x1]
        a = ('%.3f' % (round(y1, 3)))
        plt.text(x1 - 0.1, y1 + 0.005, f"{a}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means['Segway']):
        std = stds['Segway'][x1]
        a = ('%.3f' % (round(y1, 3)))
        plt.text(x1 + 0.1, y1 + 0.009, f"{a}", ha='center', va='bottom', fontsize=fontsize,
                 rotation=45, font=font, fontweight='bold')
    for x1, y1 in enumerate(means["Libbrecht's Model"]):
        std = stds["Libbrecht's Model"][x1]
        a = ('%.3f' % (round(y1, 3)))
        plt.text(x1 + 0.34, y1 + 0.005, f"{a}", ha='center', va='bottom', fontsize=fontsize,
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
    for model in ['SilenceREIN', 'ChromHMM', 'Segway', "Libbrecht's Model"]:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        fpr, tpr, mean_auc, std = average_roc(y_true_list, y_score_list)
        fpr_list[model] = fpr
        tpr_list[model] = tpr
        mean_aucs[model] = mean_auc
        stds[model] = std
    for i, model in enumerate(['SilenceREIN', 'ChromHMM', 'Segway', "Libbrecht's Model"]):
        print(f'ROC:-{model}')
        fpr = fpr_list[model]
        tpr = tpr_list[model]
        if model == 'SilenceREIN':
            a = ('%.3f' % (round(mean_aucs[model], 3)))
            b = ('%.3f' % (round(stds[model], 3)))
            plt.plot(fpr, tpr, lw=3, color=colors[i],
                     label=f'{model} (AUROC = {a}$\pm${b})')
        else:
            a = ('%.3f' % (round(mean_aucs[model], 3)))
            plt.plot(fpr, tpr, lw=3, color=colors[i],
                     label=f'{model} (AUROC = {a})')
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
    for model in ['SilenceREIN', 'ChromHMM', 'Segway', "Libbrecht's Model"]:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        precision, recall, mean_auc, std = average_pr(y_true_list, y_score_list)
        precision_list[model] = (precision)
        recall_list[model] = (recall)
        mean_auc_list[model] = (mean_auc)
        std_list[model] = (std)
    for i, model in enumerate(['SilenceREIN', 'ChromHMM', 'Segway', "Libbrecht's Model"]):
        print(f'PR:-{model}')
        recall = recall_list[model]
        precision = precision_list[model]
        if model == 'SilenceREIN':
            a = ('%.3f' % (round(mean_auc_list[model], 3)))
            b = ('%.3f' % (round(std_list[model], 3)))
            plt.plot(recall, precision, lw=2, color=colors[i],
                     label=f'{model} (AUPR = {a}$\pm${b})')
        else:
            a = ('%.3f' % (round(mean_auc_list[model], 3)))
            plt.plot(recall, precision, lw=2, color=colors[i],
                     label=f'{model} (AUPR = {a})')
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
    plt.savefig(f'Figure7.png', format='PNG', dpi=300)
    plt.show()


if __name__ == '__main__':
    value2 = {}
    filenames_SilenceREIN = [
        'data/SilenceREIN-K562-ChIA-PET-0.json',
        'data/SilenceREIN-K562-ChIA-PET-1.json',
        'data/SilenceREIN-K562-ChIA-PET-2.json',
        'data/SilenceREIN-K562-ChIA-PET-3.json',
        'data/SilenceREIN-K562-ChIA-PET-4.json',
    ]
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(filenames_SilenceREIN)
    value2['SilenceREIN'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_SilenceREIN, values_list_SilenceREIN = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_SilenceREIN)


    filenames_ChromHMM = [
        'data/Figure7/ChromHMM-score.txt',
    ]
    print('ChromHMM----------------------------------------------------')
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score_of_annotation(filenames_ChromHMM)
    value2['ChromHMM'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_ChromHMM, values_list_ChromHMM = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_ChromHMM)


    filenames_Segway = [
        'data/Figure7/Segway-score.txt',
    ]
    print('segway2----------------------------------------------------')
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score_of_annotation(filenames_Segway)
    value2['Segway'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_Segway, _values_list_Segway = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_Segway)


    t = os.getcwd()
    print(t)
    filenames_Auto = [
        'data/Figure7/FullyAutomated-score.txt',
    ]
    print('Auto----------------------------------------------------')
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score_of_annotation(filenames_Auto)
    value2["Libbrecht's Model"] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_Auto, values_list_Auto = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    print(values_Auto)

    values = {'SilenceREIN': values_SilenceREIN,
              'ChromHMM': values_ChromHMM,
              'Segway': values_Segway,
              "Libbrecht's Model": values_Auto, }

    value3 = {'SilenceREIN': values_list_SilenceREIN,
              'ChromHMM': values_list_ChromHMM,
              'Segway': _values_list_Segway,
              "Libbrecht's Model": values_list_Auto,}

    figure3(values, value2, value3)
