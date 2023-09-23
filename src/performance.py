from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import torch
import numpy as np


def evaluate(out, y_true):
    """
    compute the 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC', 'F1-score',
    'AUC-ROC', 'AUC-PR'.
    :param out: the outputs of model
    :param y_true: the real labels of samples
    :return: confusion_matrix, evaluate_results
    """
    y_pred = torch.max(out, 1)[1]
    # for 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC'
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
    result = {'Accuracy': accuracy, 'Precision': precision, 'Sensitivity': sensitivity,
              'Specificity': specificity, 'MCC': mcc}
    # for 'F1-score', 'AUC-ROC', 'AUC-PR'
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_score = out.cpu().numpy()[:, 1]
    result['F1-score'] = f1_score(y_true, y_pred)
    result['AUC-ROC'] = roc_auc_score(y_true, y_score)
    result['AUC-PR'] = average_precision_score(y_true, y_score)
    return confusion_matrix, result



def evaluate_for_cross_validation(out_and_y_true, auc_roc_filepath, auc_pr_filepath, device, draw=True):
    """
    evaluate for k-fold cross-validation
    :param out_and_y_true: a list, the out of the model and the labels of samples
    :param auc_roc_filepath:
    :param auc_pr_filepath:
    :param device: gpu or cpu
    :return: confusion_matrix, result
    """
    out = []
    y_true = []
    for j in out_and_y_true:
        t = out_and_y_true[j]
        out.extend(t['pred'])
        y_true.extend(t['real'])
    y_true = np.array(y_true)
    y_score = np.array(out)[:, 1]
    if draw:
        # for AUC-ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, color='b', label='ROC (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(auc_roc_filepath, format='SVG', dpi=1200)
        plt.show()
        # for AUC-PR
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        plt.figure()
        plt.plot(recall, precision, lw=2, color='b', label='AUPR (AUC = %0.2f)' % average_precision)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(auc_pr_filepath, format='SVG', dpi=1200)
        plt.show()
    out = torch.tensor(out, dtype=torch.float).to(device)
    y_true = torch.tensor(data=y_true, dtype=torch.int64).to(device)
    confusion_matrix, result = evaluate(out, y_true)
    return confusion_matrix, result
