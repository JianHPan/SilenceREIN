import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from common import load_json
import os
import torch
from umap import UMAP

import argparse

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def mean_dis_(data_list):
    x = []
    y = []
    for value in data_list:
        x.append(value[0])
        y.append(value[1])
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    dis_pow = 0
    for i in range(len(data_list)):
        dis_pow += (x[i] - x_mean) * (x[i] - x_mean) + (y[i] - y_mean) * (y[i] - y_mean)
    dis_pow /= len(data_list)
    return dis_pow, x_mean, y_mean


def cal(data, label):
    data_silencer = []
    data_negative = []
    data = list(data)
    label = list(label)
    for i in range(len(label)):
        if label[i] == 0:
            data_silencer.append(list(data[i]))
        if label[i] == 1:
            data_negative.append(list(data[i]))
    print(len(data_silencer) + len(data_negative))

    s_silencer, x_mean_silencer, y_mean_silencer = mean_dis_(data_silencer)
    s_negative, x_mean_negative, y_mean_negative = mean_dis_(data_negative)

    ans = (s_silencer + s_negative) / (
            (x_mean_silencer - x_mean_negative) * (x_mean_silencer - x_mean_negative)
            +
            (y_mean_silencer - y_mean_negative) * (y_mean_silencer - y_mean_negative)
    )
    return ans


def visual(x):
    # Configure UMAP hyperparameters
    reducer = UMAP(n_neighbors=15,
                   n_components=2,
                   metric='euclidean',
                   n_epochs=1000,
                   learning_rate=15.0,
                   init='random',
                   min_dist=0.1,
                   spread=1.0,
                   low_memory=False,
                   set_op_mix_ratio=1.0,
                   local_connectivity=1.0,
                   repulsion_strength=1.0,
                   negative_sample_rate=5,
                   transform_queue_size=4.0,
                   a=None,
                   b=None,
                   random_state=25,
                   metric_kwds=None,
                   angular_rp_forest=False,
                   target_n_neighbors=-1,
                   transform_seed=42,
                   verbose=False,
                   unique=False,
                   )
    # Fit and transform the data
    x_ts = reducer.fit_transform(x)

    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final


def plot_labels(s_low_d_weights, ture_labels, number_of_class):
    maker = ['o', 'o', 'o', 'o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#e38c7a', 'cyan', '#656667', '#99a4bc', 'blue', 'lime', 'r',
        'violet',
        'm', 'peru',
        'olivedrab',
        'hotpink']
    ture_labels = ture_labels.reshape((-1, 1))
    s_data = np.hstack((s_low_d_weights, ture_labels))
    s_data = pd.DataFrame({'x': s_data[:, 0], 'y': s_data[:, 1], 'label': s_data[:, 2]})

    labels = ['silencer', 'non-silencer', 'enhancer', 'promoter']
    for index in range(number_of_class):
        x = s_data.loc[s_data['label'] == index]['x']
        y = s_data.loc[s_data['label'] == index]['y']
        plt.scatter(x, y, cmap='brg', s=100, marker=maker[index], c=colors[index],
                    edgecolors=colors[index],
                    label=labels[index])

    plt.rcParams.update({'font.size': 16})
    legend = plt.legend()
    legend.get_texts()[0].set_x(-14)
    legend.get_texts()[1].set_x(-14)
    plt.legend(loc="upper right", prop={'family': font, 'size': 14}, bbox_to_anchor=(1.0, 1.0))

    plt.xticks(fontname=font, fontsize=20)
    plt.yticks(fontname=font, fontsize=20)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])



if __name__ == '__main__':
    model_names = ['Fig6A', 'Fig6B', 'Fig6C', 'Fig6D']
    fig = plt.figure(figsize=(20, 20))
    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    loc = [221, 222, 223, 224]
    titles = ['A', 'B', 'C', 'D']
    names = ['SilenceREIN',
             'SilenceREIN-alt',
             'SilenceREIN (No network features)',
             'SilenceREIN-alt (No network features)']
    setup_seed(67)
    features = load_json(f'data/{model_names[0]}.json')
    posNum = len(features['1'])
    negNum = len(features['0'])
    print(f'{posNum}\t{negNum}')
    selectSilencerIDs = random.sample(range(posNum), 500)
    selectNegativeIDs = random.sample(range(negNum), 500)

    for idx, model_name in enumerate(model_names):
        features = load_json(f'data/{model_name}.json')

        silencers = []
        negatives = []
        for value in features['1']:
            representation, node_idx, _ = value
            silencers.append(representation)
        for value in features['0']:
            representation, node_idx, _ = value
            negatives.append(representation)
        silencer_index = [0] * len(silencers)
        negative_index = [1] * len(negatives)
        X = np.array(silencers + negatives)
        Y = np.array(silencer_index + negative_index)

        X2D = visual(X)
        print(cal(X2D, Y))
        # plt.subplot(loc[idx])
        # plt.title(f'{names[idx]}', fontsize=20)
        # plt.text(-0.2, 1.15, f'({titles[idx]})', fontsize=30, fontweight='bold')
        # plot_labels(X2D, Y, 2)

        silencers_withID = []
        negatives_withID = []
        for sample_idx in selectSilencerIDs:
            silencers_withID.append(features['1'][sample_idx])
        for sample_idx in selectNegativeIDs:
            negatives_withID.append(features['0'][sample_idx])
        silencers = []
        negatives = []
        codes = []

        for value in silencers_withID:
            representation, node_idx, code = value
            silencers.append(representation)
            if code == 8:
                codes.append(0)
            else:
                assert code == 8
        for value in negatives_withID:
            representation, node_idx, code = value
            negatives.append(representation)
            if code == 8:
                assert code != 8
                codes.append(0)
            elif code == 4:
                codes.append(1)
            elif code == 2:
                codes.append(2)
            elif code == 1:
                codes.append(3)
            else:
                exit(0)
        silencer_index = [0] * len(silencers)
        negative_index = [1] * len(negatives)
        X = np.array(silencers + negatives)
        Y = np.array(silencer_index + negative_index)
        Codes = np.array(codes)
        plt.subplot(loc[idx])
        plt.title(f'{names[idx]}', fontsize=20)
        plt.text(-0.2, 1.15, f'({titles[idx]})', fontsize=30, fontweight='bold')
        plot_labels(visual(X), Codes, 4)
    plt.savefig(f'Figure6.svg', format='SVG', dpi=1200)
    plt.show()
    plt.close()


