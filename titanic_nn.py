import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_loader import TitanicPrepare
from torch import optim


class TitanicNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.to(torch.device('cpu'))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)

        x = self.sigmoid(x)
        return x