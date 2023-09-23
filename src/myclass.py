import gzip

from torch_geometric.data import InMemoryDataset
from typing import Union, List, Tuple
import torch
import os
from utils import read_compressed_data

import numpy as np


class REIN(object):
    """
    adjacency_list: An adjacency table that stores the connection relationships between nodes.
    nodes: A list of all nodes.
    node_num: The number of nodes.
    """
    def __init__(self, adjacency_list, nodes):
        self.adjacency_list = adjacency_list
        self.nodes = nodes
        self.node_num = len(nodes)

    def get_connected_components(self):
        """
        Obtain each connected component in REIN and store the node indexes in each connected component in a list.
        :return: number of connected components, list of nodes in each connected component
        """
        num = 0
        connected_components = []
        visited = [False] * self.node_num
        for i in range(self.node_num):
            if visited[i] is False:
                q = [i]
                # Recording of nodes in the connected component
                sub_graph = [i]
                visited[i] = True
                while len(q) != 0:
                    node = q[0]
                    q.pop(0)
                    for neighbor in self.adjacency_list[node]:
                        if visited[neighbor] is False:
                            q.append(neighbor)
                            sub_graph.append(neighbor)
                            visited[neighbor] = True
                num += 1
                connected_components.append(sub_graph)
        print(f'{sum(visited)}, {self.node_num}')

        return num, connected_components


class Node(object):
    """
    The node in REIN.
    node_index: the index of node, start from 0.
    coordinate: the genomic coordinates of the node, in the format: chrom-start-end.
    cis_regulatory_elements: the cis regulatory elements in this genomic interval, in the format:
                             {'silencer': [node1, node2, ...],
                              'non-silencer': [],
                              'enhancer': [],
                              'promoter': []}
    histone_modifications: The histone ChIP-seq features for this genomic interval.
    transcription_factors: The TF ChIP-seq features for this genomic interval.
    """
    def __init__(self):
        self.node_index = 0
        self.coordinate = None
        self.cis_regulatory_elements = {}
        self.seq = None


class InMemoryREINDateset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for raw_path in self.raw_paths:
            # Read DataSet from `raw_path`.
            data = torch.load(raw_path)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class NodeFeaturesDataset:
    def __init__(self, filename, featureDim=None):
        self.x = None
        with gzip.open(filename) as f:
            line = f.readline().decode('utf-8')
            print(line)
            total = int(line.split()[0])
            self.x = [None] * total
            while True:
                idx_str = f.readline()
                if not idx_str:
                    break
                idx = int(idx_str.decode('utf-8')[1:])
                features = []
                for i in range(21):
                    feat_str = f.readline().decode('utf-8')
                    fields = feat_str.split()
                    if len(fields) != 600:
                        print('the lenght of feat is not 600')
                    feat = []
                    if featureDim is None:
                        for field in fields:
                            feat.append(float(field))
                    else:
                        if i in featureDim:
                            for field in fields:
                                feat.append(float(field))
                        else:
                            feat = [0.0] * 600
                    features.append(feat)
                features = np.array(features)
                self.x[idx] = torch.tensor(data=features, dtype=torch.float)

        self.len = len(self.x)


class NodeFeaturesDataset2:
    def __init__(self, filename):
        self.x = read_compressed_data(filename)
        self.len = len(self.x)

