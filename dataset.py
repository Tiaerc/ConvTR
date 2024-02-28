import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as sio


class HSIDataset(Dataset):
    class_idx = None

    train_idx = None
    test_idx = None

    # mode取值为: full train test
    def __init__(self, mat_path, rate, train_rate, test_rate, mode):
        super().__init__()

        self.data = torch.from_numpy(sio.loadmat(mat_path)['Lena']/255.0)

        self.total_size = 1
        self.shape_list = []
        for s in self.data.shape:
            self.total_size *= s
            self.shape_list.append(s)
        self.size = int(rate * self.total_size)

        self.mode = mode

        self.train_size = int(self.size * train_rate)
        self.test_size = self.size - self.train_size

        if HSIDataset.class_idx == None:
            HSIDataset.class_idx = random.sample(range(self.total_size), self.size)
        if HSIDataset.train_idx == None:
            HSIDataset.train_idx = random.sample(HSIDataset.class_idx, self.train_size)
        if HSIDataset.test_idx == None:
            HSIDataset.test_idx = list(set(HSIDataset.class_idx) - set(HSIDataset.train_idx))

            

        # print("break point")

    def __getitem__(self, index):
        if self.mode == "train":
            true_index = self.train_idx[index]
            
        elif self.mode == "test":
            true_index = self.test_idx[index]

        else:
            true_index = self.class_idx[index]
            
        ret = []
        ret_label = self.data
        total_size = self.total_size
        for i in range(len(self.shape_list)):
            total_size /= self.data.shape[i]
            idx = int(true_index // total_size)
            true_index = true_index % total_size
            ret.append(idx)
            ret_label = ret_label[idx]

        # ret_label = ret_label/150
        ret = torch.tensor(ret)
        return ret, ret_label

    def __len__(self):
        if self.mode == "train":
            return self.train_size
        elif self.mode == "test":
            return self.test_size
        else:
            return self.size


# if __name__ == "__main__":
#     HSI_data_train= HSIDataset('Lena_256.mat', 0.9, 0.9, 0.1, 'train')
#     print(HSI_data_train[128])
