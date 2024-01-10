import numpy as np
import torch
from torch.utils.data import Dataset


class QuantumStateGenerator(object):
    def __init__(self, file, number, qubits):
        self.data = list()
        self.file = file
        self.number = number

    def save_data(self):
        torch.save(self.data, self.file)


class QuantumStateDataset(Dataset):
    def __init__(self, file):
        data = np.load(file)
        self.data = data[:, :-1]
        self.labels = data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_vector = self.data[idx].flatten()
        return data_vector, self.labels[idx]

