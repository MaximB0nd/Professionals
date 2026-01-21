import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from titanic_dataset import TitanicPrepare
from torch import optim


class TitanicNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # CrossEntropyLoss включает в себя softmax, поэтому возвращаем logits
        return x