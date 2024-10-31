
from torch.utils.data.dataset import Dataset, T
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
import pandas as pd 

from models.neuronCluster import neuronCluster
import matplotlib.pyplot as plt
from tqdm import tqdm


class TrainDataset(Dataset):

    def __init__(self, startIndex, endIndex):
        self.sequenceLength = 6
        # self.prefix = prefix
        self.startIndex = startIndex
        self.endIndex = endIndex

        self.Vtrain = pd.read_csv(r'./Train_V.csv', delimiter= ',')
        self.Ztrain = pd.read_csv(r'./Train_Z.csv', delimiter= ',')


        self.length = self.endIndex - self.startIndex

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.counter = self.startIndex
        self.pass_no = 0
        self.class_name = 'plain'
        self.iter_no = 1
        self.clusters = []
        for _ in range(16):
            self.clusters.append(neuronCluster())

    def __getitem__(self, index):
        index += self.startIndex
        data_dict = {}
        # inp_dict = {}
        seq_len = 128
        check = 0
        t_dash = index
        classname = self.Vtrain[index:index+1]['Class'].item()
        iter_no = self.Vtrain[index:index+1]['iter_No'].item()
        pass_no = self.Vtrain[index:index+1]['Pass_No'].item()

        for i in range(seq_len):
            m = index - i
            if (self.Vtrain[m:m+1]['Class'].item() != classname or \
                self.Vtrain[m:m+1]['iter_No'].item() != iter_no or \
                self.Vtrain[m:m+1]['Pass_No'].item() != pass_no):
                t_dash = m
                check = 1
                break

            if m==0:
                t_dash = m
                check = 1
                break

        for key in self.Vtrain.keys():
            if key.isnumeric():
                if check:
                    temp_v = np.array([2.44]*(seq_len - index + t_dash))
                    temp_v = np.hstack((temp_v, np.array(self.Vtrain[key].to_list()[t_dash+1:index+1])))
                    
                    cluster_v = (self.clusters[int(key)].runCluster(temp_v) + 100)/131
                    data_dict['v'+key] = cluster_v
                    data_dict['z'+key] = (np.array([self.Ztrain[key].to_numpy()[index]]) + 2.5)/5
                else:
                    temp_v = np.array(self.Vtrain[key].to_numpy()[index-seq_len+1:index+1])
                    cluster_v = (self.clusters[int(key)].runCluster(temp_v) + 100)/131
                    data_dict['v'+key] = cluster_v
                    data_dict['z'+key] = (np.array([self.Ztrain[key].to_numpy()[index]]) + 2.5)/5

        return data_dict
    def __len__(self):
        return self.length
    
class custom_data(pl.LightningDataModule):
    def setup(self, stage = None):
        self.cpu = 0
        self.pin = True
        self.batchsize = 10
        print('Loading Dataset ...')
    
    def train_dataloader(self):
        dataset = TrainDataset(0,30000)
        return DataLoader(dataset, batch_size=self.batchsize,
                          num_workers=self.cpu, shuffle=True, pin_memory=self.pin)

    def val_dataloader(self):
        dataset = TrainDataset(20000,22000)
        return DataLoader(dataset, batch_size=self.batchsize,
                          num_workers=self.cpu, shuffle=False, pin_memory=self.pin)


if __name__ == "__main__":
    data = TrainDataset(0, 50)
    # k = 30
    # for i in tqdm(range(0,50), total=50):
    #     print(data[i]['v1'].shape)
        # if data[i]['v1'].shape != (3,200):
        #     print(i)


    # print(dataset.train_dataloader())