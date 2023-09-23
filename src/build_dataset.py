import os
import pickle
import random
import argparse
import torch
from torch_geometric.data import Data
from myclass import InMemoryREINDateset

import pyBigWig as bw

from utils import read_pickle_data, save_graph_to_txt, save_node_feature, save_as_pickle


# (1) Priority: silencer, non-silencer, enhancer, promoter
# (2) If there is one and only one type of CRE inside the anchor,
#     replace the data of the anchor with the data of the first CRE
# (3) Topology and 1D features are stored separately
# (4) 1D features are stored in a list format.
#     Only the features of the nodes used are stored, the rest are filled with None.


def under_sampling(codes, y):
    silencer_index = []
    non_silencer_index = []
    enhancer_index = []
    promoter_index = []
    unlabeled_node_index = []
    for i, code in enumerate(codes):
        if 8 & code == 8:
            silencer_index.append(i)
        elif 4 & code == 4:
            non_silencer_index.append(i)
        elif 2 & code == 2:
            enhancer_index.append(i)
        elif 1 & code == 1:
            promoter_index.append(i)
        else:
            unlabeled_node_index.append(i)

    print(f'before sampling: '
          f'silencer: {len(silencer_index)}, non-silencer: {len(non_silencer_index)}, '
          f'enhancer: {len(enhancer_index)}, promoter: {len(promoter_index)}, '
          f'unlabeled: {len(unlabeled_node_index)}')
    # Number of promoters sampled
    number = round(len(promoter_index) * 0.1)
    if number <= 0:
        selected_promoter_index = []
    else:
        selected_promoter_index = random.sample(promoter_index, number)
    print(f'after sampling: '
          f'silencer: {len(silencer_index)}, non-silencer: {len(non_silencer_index)}, '
          f'enhancer: {len(enhancer_index)}, promoter: {len(selected_promoter_index)}')

    mask_index = torch.tensor([*silencer_index, *enhancer_index, *non_silencer_index, *selected_promoter_index],
                              dtype=torch.long)
    mask = torch.tensor(data=[False] * len(codes), dtype=torch.bool)
    mask.index_put_((mask_index,), torch.tensor(data=True, dtype=torch.bool))

    t2 = []
    for i in range(len(mask)):
        if mask[i]:
            if y[i] == 1:
                t2.append(1)
            else:
                t2.append(0)
        else:
            t2.append(-1)
    t2 = torch.tensor(t2, dtype=torch.long)
    a = [i for i in t2 if i.item() == 1]
    b = [i for i in t2 if i.item() == 0]
    print(f'The number of positive and negative samples after sampling are： {len(a)}, {len(b)}, respectively.')
    return mask, t2


def load_bigwig(filename, adir):
    dataset = {}
    with open(filename) as f:
        for line in f.readlines():
            values = line.split()
            if values[1] == '-':
                dataset[values[0]] = None
            else:
                print(f'{adir}/{values[1]}')
                dataset[values[0]] = bw.open(f"{adir}/{values[1]}.bigWig")
        f.close()
    return dataset


def one_hot(sequence):
    one_hot_code = []
    sequence = sequence.split()[0]
    sequence = sequence.upper()
    for nucleotide in sequence:
        if nucleotide == 'A':
            one_hot_code.append([1, 0, 0, 0])
        elif nucleotide == 'T':
            one_hot_code.append([0, 1, 0, 0])
        elif nucleotide == 'C':
            one_hot_code.append([0, 0, 1, 0])
        elif nucleotide == 'G':
            one_hot_code.append([0, 0, 0, 1])
        else:
            one_hot_code.append([0, 0, 0, 0])
    code = list(map(list, zip(*one_hot_code)))
    return code


def get_signal_p_values(bigwig_files, chrom, start, end):
    p_values = []
    for cls in bigwig_files:
        if bigwig_files[cls] is None:
            values = [0] * 600
        else:
            values = [0] * 600
            try:
                values = bigwig_files[cls].values(chrom, start, end)
            except Exception as e:
                with open('problems_in_build_REIN.txt', 'a') as f:
                    print(f'error in get_signal_p_values, {e}: {chrom}-{start}-{end}')
                    f.write(f'error in get_signal_p_values, {cls}, {e}: {chrom}-{start}-{end}')
                    f.close()
        p_values.append(values)
    return p_values



def count_ones(n):
    count = 0
    while n > 0:
        count += n & 1
        n >>= 1
    return count


code2CREClass = {1: 'promoter',
                 2: 'enhancer',
                 4: 'non-silencer',
                 8: 'silencer'}


def get_x(nodes, nodes_index_in_connected_component, codes):
    """
    Get the x, node index, coordinates for each node in this connected component
    :param nodes: a list of the compressed node data. See class 'Node' in 'myclass.py'
    :param nodes_index_in_connected_component: a list of nodes index in this connected component
    :param codes: a list of code for each node in this connected component
    :return: x, node_idx, coordinates
    """
    x = []
    nodes_idx = []
    coordinates = []
    for i, idx in enumerate(nodes_index_in_connected_component):
        node = nodes[idx]
        code = codes[i]
        if code != 0:
            key = code2CREClass[code]
            print(f'convert to {key}, the num of elements in this anchors is {len(node.cis_regulatory_elements[key])}')
            node = node.cis_regulatory_elements[key][0]
        coordinate = node.coordinate
        x.append(1)
        nodes_idx.append(idx)
        chrom, start, end = coordinate.split('-')
        chrom = chrom[3:]
        if chrom in ['y', 'x']:
            print(chrom)
        if chrom == 'X' or chrom == 'x':
            coordinates.append([-1, int(start), int(end)])
        elif chrom == 'Y' or chrom == 'y':
            coordinates.append([-2, int(start), int(end)])
        else:
            coordinates.append([int(chrom), int(start), int(end)])
    return x, nodes_idx, coordinates


def get_y(nodes, nodes_index_in_connected_component):
    """
    Get the y, codes for each node in this connected component
    :param nodes: a list of the compressed node data. See class 'Node' in 'myclass.py'
    :param nodes_index_in_connected_component: a list of nodes index in this connected component
    :return: y, the label of each node
             mask, the mask of each node
             codes, '1000', '0100', '0010', '0001' and '0000' mean
             'silencer', 'non-silencer', 'enhancer', 'promoter' and 'unknown', respectively.
    """
    y = []
    mask = []
    codes = []
    for idx in nodes_index_in_connected_component:
        node = nodes[idx]
        # Binary representation
        # Priority: silencer、non-silencer、enhancer、promoter
        bits = 0
        bits += (1 & bool(node.cis_regulatory_elements['silencer']))
        bits <<= 1
        bits += (1 & bool(node.cis_regulatory_elements['non-silencer']))
        bits <<= 1
        bits += (1 & bool(node.cis_regulatory_elements['enhancer']))
        bits <<= 1
        bits += (1 & bool(node.cis_regulatory_elements['promoter']))
        # Assigning positive and negative sample labels according to priority
        if 8 & bits == 8:
            y.append(1)
            mask.append(True)
            codes.append(8)
        elif 4 & bits == 4:
            y.append(0)
            mask.append(True)
            codes.append(4)
        elif 2 & bits == 2:
            y.append(0)
            mask.append(True)
            codes.append(2)
        elif 1 & bits == 1:
            y.append(0)
            mask.append(True)
            codes.append(1)
        else:
            y.append(-1)
            mask.append(False)
            codes.append(0)
    return y, mask, codes


def get_edge_index(adjacency_list, nodes_index_in_connected_component):
    """
    Reindex the nodes starting from 0 and construct the edge_index based on the new node indices.
    :param adjacency_list: adjacency list of REIN, see class 'REIN' in myclass.py
    :param nodes_index_in_connected_component: A list of nodes index in this connected component
    :return: edge_index
    """
    idx2idx = {}
    count = 0
    for idx in nodes_index_in_connected_component:
        idx2idx[idx] = count
        count += 1
    edge_index = []
    for idx in nodes_index_in_connected_component:
        for neighbor_idx in adjacency_list[idx]:
            edge_index.append([idx2idx[idx], idx2idx[neighbor_idx]])
    return edge_index

def construct(net, filepath):
    """
    Constructing datasets from REIN.
    :param net: REIN, see class 'REIN' in myclass.py.
    :param filepath: the folder where the data is stored.
    :return:
    """
    # Obtain the number of connected components, the indices of nodes in each connected component
    num, connected_components = net.get_connected_components()
    print(f'number of connected_components: {num}')
    for idx in range(num):
        # Node id in the connectivity component
        nodes_index_in_connected_component = connected_components[idx]
        # Only connectivity components with nodes greater than 100 are retained
        if len(nodes_index_in_connected_component) < 100:
            continue

        y, mask, codes = get_y(net.nodes, nodes_index_in_connected_component)
        # If there are no nodes with known labels, no conversion is performed
        print(f'construct for connected_components {idx}')
        x, nodes_idx, coordinates = get_x(net.nodes, nodes_index_in_connected_component, codes)
        edge_index = get_edge_index(net.adjacency_list, nodes_index_in_connected_component)

        # tensor
        mask, y = under_sampling(codes, y)

        x = torch.tensor(data=x, dtype=torch.long)
        nodes_idx = torch.tensor(data=nodes_idx, dtype=torch.long)
        coordinates = torch.tensor(data=coordinates, dtype=torch.int64)
        edge_index = torch.tensor(data=edge_index, dtype=torch.long)
        codes = torch.tensor(data=codes, dtype=torch.int64)
        graph_idx = torch.tensor(data=idx, dtype=torch.int64)

        graph = Data(x=x, y=y, nodes_idx=nodes_idx, edge_index=edge_index.t().contiguous(),
                     coordinates=coordinates,
                     mask=mask,
                     codes=codes,
                     graph_idx=graph_idx)
        torch.save(graph, f'{filepath}/{idx}.pt')

        save_graph_to_txt(graph, f'{filepath}-txt')


def build_labeled_nodes_features(nodes, graph, chip_seq_set):
    """
    Construct 1D features for labelled nodes and compress them for storage
    :param nodes: a list of the compressed node data. See class 'Node' in 'myclass.py'
    :param graph: the dataset loaded by InMemoryREINDateset. See class 'InMemoryREINDateset' in 'myclass.py'
    :param chip_seq_set: the ChIP-seq data set
    :return:
    """
    x_labeled = [None] * len(nodes)

    count = 0
    print(f'number of connected components is {len(graph)}')

    if cell == 'K562':
        # histone
        hm_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/histone-ChIP-seq-mapping.txt'
                                , f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/histone-ChIP-seq')
        # TF
        tf_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/TF-ChIP-seq-mapping.txt',
                                f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/TF-ChIP-seq')
    elif cell == 'HepG2':
        # histone
        hm_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/histone-ChIP-seq-mapping.txt'
                                , f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/histone-ChIP-seq')
        # TF
        tf_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/TF-ChIP-seq-mapping.txt',
                                f'{work_dir}/data/ChIP-seq/{cell}/{chip_seq_set}/TF-ChIP-seq')
    else:
        exit(0)


    for connected_component in graph:
        nodes_idx = connected_component.nodes_idx.tolist()
        mask = connected_component.mask.tolist()
        codes = connected_component.codes.tolist()
        coordinates = connected_component.coordinates.tolist()
        print(f'{len(codes)}\t{len(coordinates)}')
        for i in range(len(nodes_idx)):
            if mask[i] is True:
                # Get the index of the node in REIN
                idx = nodes_idx[i]
                node = nodes[idx]
                code = codes[i]
                
                coordinate = coordinates[i]
                chrom, start, end = coordinate
                if chrom == -1:
                    chrom = 'X'
                new_coordinate = f'chr{chrom}-{start}-{end}'
                if code != 0:
                    node_list = node.cis_regulatory_elements[code2CREClass[code]]


                    # check
                    key = code2CREClass[code]
                    node2 = node.cis_regulatory_elements[key][0]

                    for tmp_node in node_list:
                        if tmp_node.coordinate == new_coordinate:
                            print(f'convert node from {node.coordinate}, to ---> {tmp_node.coordinate}')
                            node = tmp_node
                            count += 1
                            break

                    # if node2 == node:
                    #     print('equal')
                    #
                    # # assert 1 == 2
                    # assert node2.seq == node.seq
                    # assert node2.coordinate == node.coordinate

                chrom, start, end = node.coordinate.split('-')
                start = int(start)
                end = int(end)
                histone_modifications = get_signal_p_values(hm_bigwig, f'{chrom}', start, end)
                transcription_factors = get_signal_p_values(tf_bigwig, f'{chrom}', start, end)

                features = [*one_hot(node.seq), *histone_modifications, *transcription_factors]
                x_labeled[idx] = features

    if chip_seq_set == 'Set1':
        save_node_feature(f'{to_dir}/x-labeled.txt.gz', x_labeled)
    if chip_seq_set == 'Set2':
        save_node_feature(f'{to_dir}/x-labeled-alt.txt.gz', x_labeled)
    print(f'number of labeled nodes is {count}')



def build_unlabeled_node_features(nodes, graph):
    """
    Construct 1D features for all unlabelled nodes and compress them for storage.
    To avoid individual files from occupying too much memory, we store node features in units of connected components.
    :param nodes: a list of the compressed node data. See class 'Node' in 'myclass.py'
    :param graph:
    :return:
    """
    # histone
    hm_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/K562/Set1/histone-ChIP-seq-mapping.txt'
                            , f'{work_dir}/data/ChIP-seq/K562/Set1/histone-ChIP-seq')
    # TF
    tf_bigwig = load_bigwig(f'{work_dir}/data/ChIP-seq/K562/Set1/TF-ChIP-seq-mapping.txt',
                            f'{work_dir}/data/ChIP-seq/K562/Set1/TF-ChIP-seq')

    # # histone
    # hm_bigwig = load_bigwig(f'/root/autodl-tmp/code/SilenceREIN0903/data/ChIP-seq/Set1/histone-ChIP-seq-mapping.txt'
    #                         , f'/root/autodl-tmp/code/SilenceREIN0903/data/ChIP-seq/Set1/histone-ChIP-seq')
    # # TF
    # tf_bigwig = load_bigwig(f'/root/autodl-tmp/code/SilenceREIN0903/data/ChIP-seq/Set1/TF-ChIP-seq-mapping.txt',
    #                         f'/root/autodl-tmp/code/SilenceREIN0903/data/ChIP-seq/Set1/TF-ChIP-seq')

    count = 0
    print(f'number of connected components is {len(graph)}')
    for i, connected_component in enumerate(graph):
        # if i==0 or i > 3:
        #     continue
        nodes_idx = connected_component.nodes_idx.tolist()
        mask = connected_component.mask.tolist()
        codes = connected_component.codes.tolist()
        # unlabeled_nodes_idx = [nodes_idx[i] for i in range(len(mask)) if not mask[i]]
        unlabeled_nodes_idx = [nodes_idx[i] for i in range(len(mask)) if ((mask[i] is False) and (codes[i] == 0))]
        unlabeled_nodes_idx2 = [nodes_idx[i] for i in range(len(mask)) if (codes[i] == 0)]
        selected_unlabeled_nodes_idx = unlabeled_nodes_idx
        print(f'number of selected unlabeled nodes in this time is {len(selected_unlabeled_nodes_idx)}')
        x_unlabeled_subgraph = [None] * len(nodes)
        for j, idx in enumerate(selected_unlabeled_nodes_idx):

            node = nodes[idx]
            chrom, start, end = node.coordinate.split('-')
            if chrom == -1:
                chrom = 'X'
            count += 1
            start = int(start)
            end = int(end)

            histone_modifications = get_signal_p_values(hm_bigwig, chrom, start, end)
            transcription_factors = get_signal_p_values(tf_bigwig, chrom, start, end)

            features = [*one_hot(node.seq), *histone_modifications, *transcription_factors]
            x_unlabeled_subgraph[idx] = torch.tensor(data=features, dtype=torch.float)

        save_as_pickle(f'{to_dir}/x-unlabeled-{i}.pkl', x_unlabeled_subgraph, True)
    print(f'number of selected unlabeled nodes is {count}')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='input your info')
    parse.add_argument('--cell', type=str, default='K562')
    parse.add_argument('--HiC', type=str, default='ChIA-PET')
    args = parse.parse_args()
    work_dir = os.getcwd()

    cell = args.cell
    HiC = args.HiC

    from_file = f"{work_dir}/data/REIN/{cell}/graph-{HiC}.pkl"
    to_dir = f"{work_dir}/dataset/{cell}/{HiC}"

    random.seed(123)

    print(from_file)

    G = read_pickle_data(from_file)

    t = G.nodes[0]
    num_of_connected_components, _ = G.get_connected_components()
    print(f'number of connected components is : {num_of_connected_components}')

    if not os.path.isdir(f'{to_dir}/raw'):
        os.makedirs(f'{to_dir}/raw')
    if not os.path.isdir(f'{to_dir}/raw-txt'):
        os.makedirs(f'{to_dir}/raw-txt')

    construct(G, f'{to_dir}/raw')

    dataset = InMemoryREINDateset(to_dir)
    build_labeled_nodes_features(G.nodes, dataset, 'Set1')
    build_labeled_nodes_features(G.nodes, dataset, 'Set2')
    # build_unlabeled_node_features(G.nodes, dataset)

