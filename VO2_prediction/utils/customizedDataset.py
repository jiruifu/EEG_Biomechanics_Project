import torch
from torch.utils.data import Dataset

class Custom3dDataset(Dataset):
    # Convert data to tensors and ensure correct dtype
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data).float().unsqueeze(1) #add one more channel dimension
        self.labels = torch.from_numpy(labels).float()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class Custom2dDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]