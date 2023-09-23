import gzip
import pickle

import torch
import numpy as np
import pandas as pd
import json
import random
import os

from tqdm import tqdm

from torch_geometric.utils import k_hop_subgraph, degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt


def record_samples(sample_ids, out, y_true, train_or_valid):
    """
    recording the sample ids, the output of model and the real label of samples a certain fold of cross-validation,
    for training dataset, the format is ['id, real_label', 'id, real_label', ...]
    for testing dataset, the format is ['id, real_label, predictive_label', 'id, real_label, predictive_label', ...]
    :param sample_ids: the ids of all samples
    :param out: outputs of the model in a certain fold of cross-validation for all samples,
    e.g. for sample a, the output is [0.5,0.8]
    :param y_true: the real labels of all samples in a certain fold of cross-validation
    :param train_or_valid: training or testing
    :return: a list
    """
    y_pred = torch.max(out, 1)[1]
    y_pred = y_pred.tolist()
    y_true = y_true.tolist()
    if train_or_valid == 'valid':
        sample_id_y_pred = [f'{a},{b},{c}' for a, b, c in zip(sample_ids, y_true, y_pred)]
    elif train_or_valid == 'train':
        sample_id_y_pred = [f'{a},{b}' for a, b in zip(sample_ids, y_true)]
    else:
        sample_id_y_pred = None
    return sample_id_y_pred


def save_samples_record(samples_record, filepath):
    """
    save the record of each epoch of K-fold cross-validation to CSV file,
    the content of the CSV file will be:
    epoch=0-fold=0-train	'id, real_label'    'id, real_label' ...
    epoch=0-fold=0-valid	'id, real_label, predictive_label'  'id, real_label, predictive_label' ...
    epoch=1-fold=0-train    ...
    epoch=1-fold=0-valid    ...
    ...
    epoch=0-fold=4-train    ...
    epoch=0-fold=4-valid    ...
    ...
    epoch=149-fold=4-train    ...
    epoch=149-fold=4-valid    ...

    :param samples_record: the records need to be saved
    :param filepath: the path to save file
    :return:
    """
    max_length = 0
    for key in samples_record:
        if len(samples_record[key]) > max_length:
            max_length = len(samples_record[key])
    tmp = []
    names = []
    for key in samples_record:
        names.append(key)
        add_length = max_length - len(samples_record[key])
        samples_record[key].extend([''] * add_length)
        tmp.append(samples_record[key])
    df = pd.DataFrame(index=names, data=tmp)
    df.to_csv(filepath)


def save_as_json(dataset, filepath):
    context = json.dumps(dataset, sort_keys=False, indent=4, separators=(',', ': '))
    with open(filepath, 'w') as save_f:
        save_f.write(context)
        save_f.close()


def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        f.close()
    return data


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_by_gzip(filename, data):
    strings = pickle.dumps(data)
    compressed_strings = gzip.compress(strings)
    with open(f'{filename}.gz', 'wb') as gzf:
        gzf.write(compressed_strings)
        gzf.close()


def save_as_pickle(filename, data, compressed=True):
    """
    Serialise, compress (optional) and save the data to a specified folder
    :param filename: specified folder
    :param data:
    :param compressed:
    :return:
    """
    if compressed is False:
        output = open(filename, 'wb')
        strings = pickle.dumps(data)
        output.write(strings)
        output.close()
    else:
        save_by_gzip(filename, data)


def read_compressed_data(filename):
    """
    Reading the compressed data from the file
    :param filename:
    :return: Decompressed and deserialized objects
    """
    with open(filename, 'rb') as f:
        compressed_data = f.read()
    # Decompression of data
    decompressed_data = gzip.decompress(compressed_data)
    # Deserialized objects
    obj = pickle.loads(decompressed_data)
    return obj


def get_decompressed_node(compressed_node):
    """
    Obtain decompressed and deserialized data from compressed data
    :param compressed_node:
    :return:
    """
    decompressed_data = gzip.decompress(compressed_node)
    node = pickle.loads(decompressed_data)
    return node


def read_pickle_data(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
        f.close()
    return loaded_data


def save_graph_to_txt(graph, filepath):
    """
    This function is used to save a description file of the PYG data
    :param graph: The data that needs to be described
    :param filepath: The path to save
    """
    x = graph.x.tolist()
    y = graph.y.tolist()
    nodes_idx = graph.nodes_idx.tolist()
    mask = graph.mask.tolist()
    codes = graph.codes.tolist()
    graph_idx = graph.graph_idx.item()
    coordinates = graph.coordinates.tolist()
    edge_index = graph.edge_index.tolist()

    context = ''
    context += 'x\ty\tmask\tnodeID\tcode\tchrom\tstart\tend\n'
    for i in range(len(x)):
        chrom, start, end = coordinates[i]
        if chrom == -1:
            chrom = 'X'
        context += f'{x[i]}\t{y[i]}\t{mask[i]}\t{nodes_idx[i]}\t{codes[i]}\t{chrom}\t{start}\t{end}\n'
    context += 'edge index:\n'
    nodes1 = edge_index[0]
    nodes2 = edge_index[1]
    for i in range(len(nodes1)):
        context += f'{nodes1[i]}\t{nodes2[i]}\n'
    with open(f'{filepath}/{graph_idx}.txt', 'w') as f:
        f.write(context)
        f.close()


def to_string(features):
    """
    convert node feature to string.
    :param features:
    """
    context = ''
    for i, feat in enumerate(features):
        line = ''
        for value in feat:
            line += f'{value}\t'
        line += '\n'
        context += line
    return context


def save_node_feature(filename, features):
    """
    This function is used to save node features
    :param filename:
    :param features:
    """
    with gzip.open(filename, 'w') as f:
        f.write(f'{len(features)}\n'.encode('utf-8'))
        context = []
        for i in tqdm(range(len(features))):
            if features[i] is None:
                continue
            context.append(f'>{i}\n'.encode('utf-8'))
            context.append(to_string(features[i]).encode('utf-8'))
        f.writelines(context)
        f.close()


def get_dataloader(connected_component_list, batch_size, shuffle=True, dim=10):
    """
    Obtain the first-order subgraph of the target node and return it to the Dataloader
    :param connected_component_list:
    :param batch_size:
    :param shuffle:
    :return: loader
    """
    subgraph_list = []
    for connected_component in connected_component_list:

        degrees = degree(connected_component.edge_index[0])

        mask = connected_component.mask
        targets = mask.nonzero()
        edge_index = connected_component.edge_index
        for node_idx in targets:
            subset, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_idx=node_idx,
                                                                     num_hops=1,
                                                                     edge_index=edge_index,
                                                                     relabel_nodes=True,
                                                                     )
            # Get features, edge information and labels of subgraph nodes
            t = [1.0] * dim
            subgraph_x = torch.tensor([t] * len(subset), dtype=torch.float)

            subgraph_y = connected_component.y[subset]
            t2 = [1 if i.item() == 1 else 0 for i in subgraph_y]
            t2 = torch.tensor(t2, dtype=torch.long)
            subgraph_y = t2
            subgraph_nodes_idx = connected_component.nodes_idx[subset]
            subgraph_coordinates = connected_component.coordinates[subset]
            subgraph_codes = connected_component.codes[subset]
            subgraph_mask = torch.tensor(data=[False] * len(subset), dtype=torch.bool)
            subgraph_mask.index_put_((mapping,), torch.tensor(data=True, dtype=torch.bool))
            subgraph_data = Data(x=subgraph_x, y=subgraph_y,
                                 edge_index=subgraph_edge_index,
                                 nodes_idx=subgraph_nodes_idx,
                                 coordinates=subgraph_coordinates,
                                 codes=subgraph_codes,
                                 mask=subgraph_mask)
            subgraph_list.append(subgraph_data)
    print(f'Number of graphs: {len(subgraph_list)}')
    loader = DataLoader(subgraph_list, batch_size=batch_size, shuffle=shuffle)
    return loader



def get_the_number_of_positive_negative_samples(loader):
    result = {'pos': 0, 'neg': 0}
    for index, batch in enumerate(loader):
        for (label, mask) in zip(batch.y, batch.mask):
            if mask:
                label = label.item()
                if label == 1:
                    result['pos'] = result.get('pos', 0) + 1
                else:
                    result['neg'] = result.get('neg', 0) + 1
    return result


def get_feature_dim(feature):
    feature2Dim = {
        'H3K9me3': 4,
        'H3K27me3': 5,
        'H3K27ac': 6,
        'H3K36me3': 7,
        'H3K4me3': 8,
        'H3K4me1': 9,
        'H3K79me2': 10,
        'H2AFZ': 11,
        'H4K20me1': 12,
        'H3K4me2': 13,
        'H3K9ac': 14,
        'H3K9me1': 15,
        'RAD21': 16,
        'POLR2A': 17,
        'SMC3': 18,
        'ZNF143': 19,
        'CTCF': 20
    }
    featureDim = [0, 1, 2, 3]
    if feature == 'All':
        return None
    if feature not in feature2Dim.keys():
        return None
    featureDim.append(feature2Dim[feature])
    return featureDim


