import matplotlib.pyplot as plt
from common import average_roc, average_pr, get_y_ture_y_pred_y_score


def figureS1(values2):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#1a9850', '#d73027', '#fdae61', '#abd9e9', '#f46d43',
        '#d7191c', '#abdda4', '#f0f0f0', '#fed9a6', '#636363'
    ]

    plt.figure(figsize=(10, 20))

    # plt.gcf().subplots_adjust(left=0.0, right=0.6)

    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    # ROC
    plt.subplot(211)

    plt.text(-0.15, 1.10, "(A)", fontsize=20, fontweight='bold')
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
    # plt.legend(loc="lower right", prop={'family': font, })
    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2),
               loc=num3,
               borderaxespad=num4,
               prop={'family': font, 'size': 12}
               )
    plt.xticks(font=font, )
    plt.yticks(font=font, )

    # PR
    plt.subplot(212)
    plt.text(-0.15, 1.10, "(B)", fontsize=20, fontweight='bold')
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
    # plt.legend(loc="lower right")

    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2),
               loc=num3,
               borderaxespad=num4,
               prop={'family': font, 'size': 12}
               )

    plt.savefig('FigureS1.svg', format='SVG', dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    fileNameDict = {}
    fileNameDict['All'] = [
        'data/SilenceREIN-K562-ChIA-PET-0.json',
        'data/SilenceREIN-K562-ChIA-PET-1.json',
        'data/SilenceREIN-K562-ChIA-PET-2.json',
        'data/SilenceREIN-K562-ChIA-PET-3.json',
        'data/SilenceREIN-K562-ChIA-PET-4.json',
    ]

    for ChIPSEQ in [
        'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K36me3', 'H3K4me3', 'H3K4me1',
        'H3K79me2', 'H2AFZ', 'H4K20me1', 'H3K4me2', 'H3K9ac', 'H3K9me1', 'RAD21',
        'POLR2A3', 'SMC3', 'ZNF143', 'CTCF4'
    ]:
        fileNameDict[ChIPSEQ] = []
    for ChIPSEQ in [
        'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K36me3', 'H3K4me3', 'H3K4me1',
        'H3K79me2', 'H2AFZ', 'H4K20me1', 'H3K4me2', 'H3K9ac', 'H3K9me1', 'RAD21',
        'POLR2A3', 'SMC3', 'ZNF143', 'CTCF4'
    ]:
        for i in range(5):
            fileNameDict[ChIPSEQ].append(f'data/FigureS1/SilenceREIN-CV-K562-ChIP-PET-{ChIPSEQ}-{i}.json')

    value2 = {}
    for ChIPSEQ in [
        'All',
        'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K36me3', 'H3K4me3', 'H3K4me1',
        'H3K79me2', 'H2AFZ', 'H4K20me1', 'H3K4me2', 'H3K9ac', 'H3K9me1', 'RAD21',
        'POLR2A3', 'SMC3', 'ZNF143', 'CTCF4'
    ]:
        filenames = fileNameDict[ChIPSEQ]
        y_ture_list, _, y_score_list = get_y_ture_y_pred_y_score(filenames)
        if ChIPSEQ == 'POLR2A3':
            ChIPSEQ = 'POLR2A'
        if ChIPSEQ == 'CTCF4':
            ChIPSEQ = 'CTCF'
        value2[ChIPSEQ] = {'y_true_list': y_ture_list, 'y_score_list': y_score_list}
    figureS1(value2)
