import torch
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=100,
                            kernel_size=3,
                            padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3,
                               stride=3,
                               padding=0),
            torch.nn.Dropout(p=0.2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=100,
                            out_channels=100,
                            kernel_size=3,
                            padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3,
                               stride=3,
                               padding=1),
            torch.nn.Dropout(p=0.2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=100,
                            out_channels=100,
                            kernel_size=3,
                            padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3,
                               stride=3,
                               padding=1),
            torch.nn.Dropout(p=0.2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=100,
                            out_channels=100,
                            kernel_size=3,
                            padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3,
                               stride=3,
                               padding=1),
            torch.nn.Dropout(p=0.2)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return x


class SilenceREIN(torch.nn.Module):
    def __init__(self, out_channels=2):
        super(SilenceREIN, self).__init__()
        self.sage1 = SAGEConv(in_channels=10, out_channels=100, aggr="mean")
        self.cnn = CNN(21)
        self.bn1 = torch.nn.BatchNorm1d(900)
        self.fc1 = torch.nn.Linear(900, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, out_channels)

    def forward(self, x1, x2, edge_index, batch):
        x1 = self.sage1(x1, edge_index)
        x1 = global_add_pool(x1, batch)
        x2 = self.cnn(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x
        # return x.log_softmax(dim=-1)
