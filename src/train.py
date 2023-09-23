import os

from myclass import InMemoryREINDateset, NodeFeaturesDataset
from log import Logger
import argparse
from utils import setup_seed, get_dataloader, get_the_number_of_positive_negative_samples
from performance import evaluate
import torch
from model import SilenceREIN
import torch.nn.functional as F


def train(loader):
    model.train()
    total_loss = total_correct = 0
    count = 0
    for j, data in enumerate(loader):
        optimizer.zero_grad()

        nodes_idx = data.nodes_idx
        mask = data.mask

        x1 = data.x.to(device)
        x2 = torch.stack([nodes.x[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
        y = data.y[mask].to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        out = model(x1, x2, edge_index, batch)

        # # label smoothing
        # loss = ls_loss(out, y)

        weight = torch.tensor([Weight, 1.0]).to(device)
        loss = F.cross_entropy(input=out, target=y,
                               weight=weight,
                               # label_smoothing=Smoothing
                               )

        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())

        count += sum(mask)

    avg_loss = total_loss / len(loader)
    approx_acc = total_correct / int(count)
    return avg_loss, approx_acc


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='input your info')
    parse.add_argument('--model_name', type=str, default='SilenceREIN', help='the name of this model')
    parse.add_argument('--epoch', type=int, default=40, help='number of training epoch')
    parse.add_argument('--batch_size', type=int, default=128, help='number of training batch size')
    parse.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight')
    parse.add_argument('--weight', type=float, default=1.2, help='weight')
    args = parse.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_seed(12)
    work_dir = os.getcwd()
    # parse
    flag = args.model_name
    Dataset = f'{work_dir}/dataset/K562/ChIA-PET'
    Epoch = args.epoch
    Batch_size = args.batch_size
    LR = args.lr
    WeightDecay = args.weight_decay
    Weight = args.weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = Logger(f'/result/logs/{flag}').logger
    logger.info(f'{LR}')
    logger.info(f'flag: {flag}')
    logger.info(f'weight = {Weight}')
    logger.info(f'batch size: {Batch_size}')
    logger.info(f'epoch: {Epoch}')

    # dataset
    dataset = InMemoryREINDateset(Dataset)
    nodes = NodeFeaturesDataset(f'{Dataset}/x-labeled.txt.gz')

    logger.info(dataset)
    dataset, perm = dataset.shuffle(return_perm=True)

    train_loader = get_dataloader(dataset, Batch_size, shuffle=True)

    logger.info(f'Learn rate: {LR}')
    logger.info(f'Weight decay: {WeightDecay}')

    model = SilenceREIN()

    model = model.to(device)
    logger.info(f'{model}')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WeightDecay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WeightDecay,
    #                             momentum=0.3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    loss_last = float('inf')
    for epoch in range(Epoch):
        logger.info(f'{epoch}:')
        loss_, train_acc = train(train_loader)
        logger.info(f'train: {loss_}, {train_acc}')
        if loss_ > loss_last:
            logger.info('decrease learning rate ... ')
            scheduler.step()
        loss_last = loss_

    # save model
    torch.save(model.state_dict(), f'{work_dir}/result/models/{flag}.pt')

