import torch
from torch.utils.data import Dataset

class GetData(Dataset):
    def __init__(self, datax, labely, weights_input, device) -> None:
        self.data = torch.from_numpy(datax.astype(float)).to(device)
        self.label = torch.from_numpy(labely.astype(float)).to(device)
        self.weight = torch.from_numpy(weights_input.astype(float)).to(device)
        self.len = len(datax)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.weight[index]

    def __len__(self):
        return self.len