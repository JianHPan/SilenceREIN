import numpy as np
import json
import torch
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        f.close()
    return data


def get_y_ture_y_pred_y_score(filenames):
    y_ture_list = []
    y_score_list = []
    y_pred_list = []
    for filename in filenames:
        out_and_y_true = load_json(filename)
        out = []
        y_true = []
        for fold in out_and_y_true:
            t = out_and_y_true[fold]
            out.extend(t['pred'])
            y_true.extend(t['real'])
        y_true = np.array(y_true)
        y_score = np.array(out)[:, 1]
        out = torch.tensor(out, dtype=torch.float)
        y_pred = torch.max(out, 1)[1]
        y_pred = y_pred.tolist()

        y_ture_list.append(y_true)
        y_score_list.append(y_score)
        y_pred_list.append(y_pred)
    return y_ture_list, y_pred_list, y_score_list


def average_roc(y_true_list, y_score_list):
    fpr_list = []
    tpr_list = []
    auc_list = []
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)
    print(auc_list)
    n = len(fpr_list)
    all_fpr = np.unique(np.concatenate([fpr_list[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= n
    return all_fpr, mean_tpr, float(np.mean(auc_list)), float(np.std(auc_list))


def average_pr(y_true_list, y_score_list):
    precision_list = []
    recall_list = []
    auc_list = []
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall, precision)
        recall_list.append(recall)
        precision_list.append(precision)
        auc_list.append(auc_pr)
    print(auc_list)
    n = len(recall_list)
    all_recall = np.unique(np.concatenate([recall_list[i] for i in range(n)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n):
        recall = recall_list[i]
        precision = precision_list[i]
        reversed_recall = np.fliplr([recall])[0]
        reversed_precision = np.fliplr([precision])[0]
        temp = np.interp(all_recall, reversed_recall, reversed_precision)
        mean_precision += temp
    mean_precision /= n
    return mean_precision, all_recall, float(np.mean(auc_list)), float(np.std(auc_list))

