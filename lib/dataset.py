import torch 
from torch.utils.data import Dataset

class Audio_Dataset(Dataset):

    def __init__(self, X, y, length, transform):
        self.X = X
        self.y = y
        self.length = length 
        self.transform = transform 

    def __getitem__(self, i):
        sample = (self.X[i], self.y[i])
        if self.transform:
            return self.transform(sample) 
        return sample 

    def __len__(self):
        return self.length 
     
class Zhoujie_Wall_Dataset(Dataset):

    def __init__(self, X, y, length, transform=None):
        self.X = X
        self.y = y 
        self.length = length 
        self.transform = transform 

    def __getitem__(self, i):
        sample = (self.X[i], self.y[i])
        if self.transform:
            return self.transform(sample) 
        return sample 

    def __len__(self):
        return self.length 

class Zhoujie_Soil_Dataset(Dataset):

    def __init__(self, X, y, length, transform):
        self.X = X
        self.y = y 
        self.length = length 
        self.transform = transform 

    def __getitem__(self, i):
        sample = (self.X[i], self.y[i])
        if self.transform:
            return self.transform(sample) 
        return sample 


    def __len__(self):
        return self.length 
     