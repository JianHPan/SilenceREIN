import os

import torch
from myclass import InMemoryREINDateset, NodeFeaturesDataset, NodeFeaturesDataset2
from model import SilenceREIN
import argparse
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import setup_seed


def get_dataloader(connected_component_list, all_mask, batch_size, shuffle=True):
    subgraph_list = []
    for connected_component in connected_component_list:
        mask = [all_mask[idx].item() for idx in connected_component.nodes_idx]
        mask = torch.tensor(mask, dtype=torch.bool)
        print(f'number of unlabeled node is {sum(mask)}')
        targets = mask.nonzero()
        edge_index = connected_component.edge_index
        codes = connected_component.codes
        for i, node_idx in enumerate(targets):
            code = codes[node_idx]
            if code != 0:
                continue
            # print(f'{i}, for {node_idx}')
            subset, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_idx=node_idx,
                                                                     num_hops=1,
                                                                     edge_index=edge_index,
                                                                     relabel_nodes=True)
            t = [1.0] * 10
            subgraph_x = torch.tensor([t] * len(subset), dtype=torch.float)

            subgraph_y = connected_component.y[subset]
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

    print(f'Number of sub graphs: {len(subgraph_list)}')
    loader = DataLoader(subgraph_list, batch_size=batch_size, shuffle=shuffle)
    return loader


def error_log(filepath, name, information):
    with open(f'{filepath}/error-{name}.txt', 'a') as l:
        l.write(f'{information}\n')
        l.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='input your info')
    parse.add_argument('--model_name', type=str, default='SilenceREIN', help='the path of this model')

    # setup_seed(123)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    work_dir = os.getcwd()
    args = parse.parse_args()
    model_name = args.model_name
    Dataset = f'{work_dir}/dataset/K562/ChIA-PET'
    path = f'{work_dir}/result/predict'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = InMemoryREINDateset(Dataset)

    for idx in range(0, 24):
        # nodes = NodeFeaturesDataset(f'{Dataset}/x-unlabeled.txt.gz')
        nodes = NodeFeaturesDataset2(f'{Dataset}/x-unlabeled-{idx}.pkl.gz')

        predicted_mask = [True if x is not None else False for x in nodes.x]

        predicted_mask = torch.tensor(predicted_mask, dtype=torch.bool)
        print(f'{idx}: {sum(predicted_mask)}')
        data_loader = get_dataloader(dataset, predicted_mask, 128, False)

        model = SilenceREIN()
        model.to(device)
        try:
            model.load_state_dict(torch.load(f'{work_dir}/result/models/{model_name}.pt'))
            print(f'predict using {model_name}')
        except Exception as e:
            print(f'{model_name} : {e}')

        model.eval()
        # predictive silencers
        predictive_silencers = []
        silencers_count = 0
        count = 0
        with torch.no_grad():
            for j, data in enumerate(data_loader):
                nodes_idx = data.nodes_idx
                mask = data.mask

                x1 = data.x.to(device)
                x2 = torch.stack([nodes.x[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
                y = data.y[mask].to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)
                out = model(x1, x2, edge_index, batch)
                out = torch.nn.functional.softmax(out, dim=-1)
                y_pred = out.argmax(dim=-1)
                # y_pred = torch.max(out, 1)[1]
                silencer_indexes = y_pred.bool()

                silencers_count += sum(silencer_indexes)
                count += len(silencer_indexes)
                t1 = data.coordinates
                t2 = t1[mask]
                t3 = t2[silencer_indexes]
                t4 = t3.tolist()
                predictive_silencers.extend(t4)

        print(f'{model_name}: {float(silencers_count) / float(count)}')
        # save as bed file
        strs = ''
        convert = {-1: 'X'}
        for silencer in predictive_silencers:
            chrom, start, end = silencer
            chrom = int(chrom)
            start = int(start)
            end = int(end)
            chrom = convert.get(chrom, chrom)
            strs += f'chr{chrom}\t{start}\t{end}\n'
        # save as bed file
        with open(f'{path}/predicted-silencers-{idx}.bed', 'w') as f:
            f.write(strs)
            f.close()


