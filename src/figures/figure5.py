import matplotlib.pyplot as plt
from common import average_roc, average_pr, get_y_ture_y_pred_y_score
from figure3 import evaluate, get_mean_and_std


def figure5(values2):
    """
    :param values2: {'SilenceREIN': {'': [], '': []},
                    }
    :return: None
    """
    colors = ['#4F9F3C', '#EF8D8E', '#599AAD', '#C799C7'
        , '#ee6a5b', '#f6b654', '#c7dbd5', '#4ea59f', 'teal', 'royalblue', 'darkred']
    plt.figure(figsize=(10, 5))

    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    # ROC
    plt.subplot(121)

    plt.text(-0.15, 1.10, "(A)", fontsize=15, fontweight='bold')
    fpr_list = {}
    tpr_list = {}
    mean_aucs = {}
    stds = {}
    for model in values2:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        fpr, tpr, mean_auc, std = average_roc(y_true_list, y_score_list)
        fpr_list[model] = fpr
        tpr_list[model] = tpr
        mean_aucs[model] = mean_auc
        stds[model] = std
    for i, model in enumerate(values2):
        print(f'ROC:-{model}')
        fpr = fpr_list[model]
        tpr = tpr_list[model]
        a = ('%.3f' % (round(mean_aucs[model], 3)))
        b = ('%.3f' % (round(stds[model], 3)))
        plt.plot(fpr, tpr, lw=2, color=colors[i],
                 label=f'{model} (AUROC = {a}$\pm${b})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', font=font)
    plt.ylabel('True Positive Rate', font=font)
    # plt.title('Receiver operating characteristic')
    plt.title('ROC', font=font)
    plt.legend(loc="lower right", prop={'family': font, })
    plt.xticks(font=font, )
    plt.yticks(font=font, )

    # PR
    plt.subplot(122)
    plt.text(-0.15, 1.10, "(B)", fontsize=15, fontweight='bold')
    precision_list = {}
    recall_list = {}
    mean_auc_list = {}
    std_list = {}
    for model in values2:
        y_true_list = values2[model]['y_true_list']
        y_score_list = values2[model]['y_score_list']
        precision, recall, mean_auc, std = average_pr(y_true_list, y_score_list)
        precision_list[model] = (precision)
        recall_list[model] = (recall)
        mean_auc_list[model] = (mean_auc)
        std_list[model] = (std)
    for i, model in enumerate(values2):
        print(f'PR:-{model}')
        recall = recall_list[model]
        precision = precision_list[model]
        a = ('%.3f' % (round(mean_auc_list[model], 3)))
        b = ('%.3f' % (round(std_list[model], 3)))
        plt.plot(recall, precision, lw=2, color=colors[i],
                 label=f'{model} (AUPR = {a}$\pm${b})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR')
    plt.legend(loc="lower right")
    plt.savefig('Figure5.svg', format='SVG', dpi=1200)
    plt.show()


if __name__ == '__main__':

    topological = [
        'data/SilenceREIN-TOPO-0.json',
        'data/SilenceREIN-TOPO-1.json',
        'data/SilenceREIN-TOPO-2.json',
        'data/SilenceREIN-TOPO-3.json',
        'data/SilenceREIN-TOPO-4.json',
    ]
    linear = [
        'data/Figure5/SilenceREIN-Linear-Seed-is-1.json',
        'data/Figure5/SilenceREIN-Linear-Seed-is-4.json',
        'data/Figure5/SilenceREIN-Linear-Seed-is-5.json',
        'data/Figure5/SilenceREIN-Linear-Seed-is-7.json',
        'data/Figure5/SilenceREIN-Linear-Seed-is-9.json',
    ]

    value2 = {}
    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(topological)
    value2['Topological information'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_topological, _ = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)

    y_ture_list, y_pred_list, y_score_list = get_y_ture_y_pred_y_score(linear)
    value2['Linear information'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    values_linear, _ = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)

    filenames_SilenceREIN = [
        'data/SilenceREIN-K562-ChIA-PET-0.json',
        'data/SilenceREIN-K562-ChIA-PET-1.json',
        'data/SilenceREIN-K562-ChIA-PET-2.json',
        'data/SilenceREIN-K562-ChIA-PET-3.json',
        'data/SilenceREIN-K562-ChIA-PET-4.json',
    ]
    y_ture_list, _, y_score_list = get_y_ture_y_pred_y_score(filenames_SilenceREIN)
    values_all, _, = get_mean_and_std(y_ture_list, y_pred_list, y_score_list)
    value2['SilenceREIN'] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    figure5(value2)


