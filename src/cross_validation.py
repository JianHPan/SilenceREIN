import os

from myclass import InMemoryREINDateset, NodeFeaturesDataset
from utils import record_samples, save_samples_record, save_as_json
from log import Logger
import argparse

from utils import setup_seed, get_dataloader, get_the_number_of_positive_negative_samples
from performance import evaluate, evaluate_for_cross_validation

import torch

import torch.nn.functional as F
from model import SilenceREIN


from utils import get_feature_dim


def train(loader):
    model.train()
    total_loss = total_correct = 0
    count = 0
    for j, data in enumerate(loader):
        optimizer.zero_grad()

        nodes_idx = data.nodes_idx
        mask = data.mask

        x1 = data.x.to(device)
        # x1 = torch.stack([embeddings[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
        # x1 = data.x.to(device)[mask]
        x2 = torch.stack([nodes.x[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
        # x2 = None
        y = data.y[mask].to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        out = model(x1, x2, edge_index, batch)

        weight = torch.tensor([Weight, 1.0]).to(device)
        loss = F.cross_entropy(input=out, target=y,
                               weight=weight,
                               )
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())

        count += sum(mask)

    avg_loss = total_loss / len(loader)
    approx_acc = total_correct / int(count)
    return avg_loss, approx_acc


def test(loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    count = 0
    # the out of the model
    all_out = []
    # the true label of samples
    all_label = []
    # the id of samples
    all_sample_ids = []
    with torch.no_grad():
        for j, data in enumerate(loader):
            nodes_idx = data.nodes_idx
            mask = data.mask

            x1 = data.x.to(device)
            # x1 = torch.stack([embeddings[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
            # x1 = data.x.to(device)[mask]
            x2 = torch.stack([nodes.x[idx] for i, idx in enumerate(nodes_idx) if mask[i]]).to(device)
            # x2 = None
            y = data.y[mask].to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            out = model(x1, x2, edge_index, batch)

            weight = torch.tensor([Weight, 1.0]).to(device)
            loss = F.cross_entropy(input=out, target=y,
                                   weight=weight,
                                   )

            total_loss += float(loss)
            # there have no softmax in neural network
            out = torch.nn.functional.softmax(out, dim=-1)

            total_correct += int(out.argmax(dim=-1).eq(y).sum())
            count += sum(mask)

            all_out.extend(out.tolist())
            all_label.extend(y.tolist())
            # get the id of samples
            sample_ids = nodes_idx[mask]
            all_sample_ids.extend(sample_ids.tolist())
    avg_loss = total_loss / len(loader)
    print(f'Test loss: {avg_loss}')
    approx_acc = total_correct / int(count)
    print(f'Test Acc: {approx_acc}')
    all_out = torch.tensor(data=all_out, dtype=torch.float).to(device)
    all_label = torch.tensor(data=all_label, dtype=torch.int64).to(device)
    return avg_loss, all_out, all_label, all_sample_ids


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='input your info')
    parse.add_argument('--model_name', type=str, default='SilenceREIN-K562-ChIA-PET-CV',
                       help='the name of this model')
    parse.add_argument('--dataset', type=str, default='Set1', help='the filepath of dataset')
    parse.add_argument('--epoch', type=int, default=40, help='number of training epoch')
    parse.add_argument('--batch_size', type=int, default=128, help='number of training batch size')
    parse.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight')
    parse.add_argument('--weight', type=float, default=1.2, help='weight')
    parse.add_argument('--seed', type=int, default=0)
    parse.add_argument('--ChIP_seq', type=str, default='All')
    args = parse.parse_args()

    ChIP_seq = args.ChIP_seq
    featureDim = get_feature_dim(ChIP_seq)
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

    print(device)

    logger = Logger(f'/result/logs/{flag}').logger
    logger.info(f'{LR}')
    logger.info(f'flag: {flag}')
    logger.info(f'weight = {Weight}')
    logger.info(f'batch size: {Batch_size}')
    logger.info(f'epoch: {Epoch}')
    logger.info(f'seed: {args.seed}')
    logger.info(f'dataset: {Dataset}')
    logger.info(f'features: {ChIP_seq}')


    # dataset
    dataset = InMemoryREINDateset(Dataset)
    if args.dataset == 'Set1':
        nodes = NodeFeaturesDataset(f'{Dataset}/x-labeled.txt.gz', featureDim=featureDim)
    elif args.dataset == 'Set2':
        nodes = NodeFeaturesDataset(f'{Dataset}/x-labeled-alt.txt.gz', featureDim=featureDim)
    else:
        exit(0)

    logger.info(dataset)
    # dataset, perm = dataset.shuffle(return_perm=True)
    logger.info(f'len(dataset) = {len(dataset)}')
    g = torch.Generator()
    perm = torch.randperm(len(dataset), generator=g.manual_seed(args.seed))
    dataset = dataset[perm]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    setup_seed(12)

    logger.info(f'torch.init_seed: {torch.initial_seed()}')
    logger.info(f'g.init_seed: {g.initial_seed()}')

    # K-fold cross-validation
    K = 5
    fold_size = len(dataset) // K
    remainder = len(dataset) % K
    start_idx = 0

    # record the predictive results and real labels of each fold
    Out_YTrue = {}
    # record the sample ids, real labels and predictive labels of each fold
    ID_YTure_YPred = {}

    for fold in range(K):
        logger.info(f'Fold: {fold}')
        end_idx = start_idx + fold_size

        if fold < remainder:
            end_idx += 1

        test_dataset = dataset[start_idx:end_idx]
        train_dataset = dataset[0:start_idx] + dataset[end_idx:]

        start_idx = end_idx

        test_loader = get_dataloader(test_dataset, Batch_size,
                                     shuffle=False,)
        train_loader = get_dataloader(train_dataset, Batch_size,
                                      shuffle=True,)
        print('train:')
        print(get_the_number_of_positive_negative_samples(train_loader))
        print('test:')
        print(get_the_number_of_positive_negative_samples(test_loader))

        logger.info(f'Learn rate: {LR}')
        logger.info(f'Weight decay: {WeightDecay}')

        model = SilenceREIN()
        torch.save(model.state_dict(), f'{work_dir}/result/models/{flag}-init-{fold}.pt')
        model = model.to(device)
        logger.info(f'{model}')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WeightDecay)
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

            # validate on training set
            _, outputs, y_true, sampleIDs = test(train_loader)
            con_matrix, res_train = evaluate(outputs, y_true)
            logger.info(f'train confusion matrix: {con_matrix}')
            logger.info(f'train evaluate result: {res_train}')
            # record: ids, predictive labels and real labels of training samples
            tmp = record_samples(sampleIDs, outputs, y_true, 'train')
            ID_YTure_YPred[f'epoch={epoch}-fold={fold}-train'] = tmp
            # validate on testing set
            loss_valid, outputs, y_true, sampleIDs = test(test_loader)
            con_matrix, res_valid = evaluate(outputs, y_true)
            logger.info(f'test/validate confusion matrix: {con_matrix}')
            logger.info(f'test/validate evaluate result: {res_valid}')
            # record: ids, predictive labels and real labels of testing samples
            tmp = record_samples(sampleIDs, outputs, y_true, 'valid')
            ID_YTure_YPred[f'epoch={epoch}-fold={fold}-valid'] = tmp

        # save model
        torch.save(model.state_dict(), f'{work_dir}/result/models/{flag}_{fold}.pt')
        # test
        test_loss, outputs, y_true, sampleIDs = test(test_loader)
        con_matrix, res = evaluate(outputs, y_true)
        logger.info(f'test confusion matrix: {con_matrix}')
        logger.info(f'test evaluate result: {res}')
        Out_YTrue[fold] = {}
        Out_YTrue[fold]['pred'] = outputs.tolist()
        Out_YTrue[fold]['real'] = y_true.tolist()

    # evaluate for K-fold cross-validation
    con_matrix, res_cv = evaluate_for_cross_validation(Out_YTrue,
                                                       f'{work_dir}/result/figures/AUC-ROC_{flag}.svg',
                                                       f'{work_dir}/result/figures/AUC-PR_{flag}.svg',
                                                       device)
    logger.info(f'Cross Valid confusion matrix: {con_matrix}')
    logger.info(f'Cross Valid evaluate result: {res_cv}')
    # save the prediction results of the model in the K-fold cross-validation
    save_as_json(Out_YTrue, f'{work_dir}/result/records/{flag}.json')
    # # save the sample ids, real labels and predictive labels
    # save_samples_record(ID_YTure_YPred, f'{work_dir}/result/records/{flag}.csv')
