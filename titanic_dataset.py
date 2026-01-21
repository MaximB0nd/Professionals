
import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_df():
    df = pd.read_csv('titanic.csv')

class TitanicDataset(Dataset):
    def __init__(self, df, target):
        self.data = df.drop(target, axis=1).astype(np.float32)
        self.target = df[target].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Возвращаем класс как целое число (0 или 1) вместо one-hot вектора
        return torch.tensor(self.data.iloc[idx], dtype=torch.float32), torch.tensor([1 if self.target.iloc[idx] == 0 else 0, 1 if self.target.iloc[idx] == 1 else 0], dtype=torch.float32)

class TitanicPrepare():
    def __init__(self, path, target):
        self.path = path
        self.target = target

    def get_data(self):
        df = pd.read_csv(self.path)

        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[self.target])

        train = TitanicDataset(train, target=self.target)
        test = TitanicDataset(test, target=self.target)

        train_loader = DataLoader(train, batch_size=16, shuffle=True)
        test_loader = DataLoader(test, batch_size=16, shuffle=True)

        return train_loader, test_loader

