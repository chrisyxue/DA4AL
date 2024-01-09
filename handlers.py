import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class MedMNIST_ThreeChannel_Handler(Dataset):
    def __init__(self, X, Y, transform, aug_samples_lst=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.transform_aug = None
        self.aug_samples_lst = aug_samples_lst
    
    def get_aug_trans(self,transform_aug):
        self.transform_aug = transform_aug

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)

        if self.transform_aug == None:
            x = self.transform(x)
            return x, y, index
        else:
            x_orign = self.transform(x)
            x_aug = self.transform_aug(x)
            return [x_orign, x_aug] ,y, index
            
    def __len__(self):
        return len(self.X)

class MedMNIST_OneChannel_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.transform_aug = None
    
    def get_aug_trans(self,transform_aug):
        self.transform_aug = transform_aug

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # print(x.shape)
        x = Image.fromarray(x, mode='L')
        if self.transform_aug == None:
            x = self.transform(x)
            return x, y, index
        else:
            x_orign = self.transform(x)
            # sprint(x.shape)
            # print(x_orign.shape)
            x_aug = self.transform_aug(x)
            return [x_orign, x_aug] ,y, index

    def __len__(self):
        return len(self.X)


class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.transform_aug = None

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)

        if self.transform_aug == None:
            x = self.transform(x)
            return x, y, index
        else:
            x_orign = self.transform(x)
            x_aug = self.transform_aug(x)
            return [x_orign, x_aug] ,y, index
    
    def get_aug_trans(self,transform_aug):
        self.transform_aug = transform_aug

    def __len__(self):
        return len(self.X)