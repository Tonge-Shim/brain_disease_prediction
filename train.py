import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import tqdm
from operator import add
import wandb
import matplotlib.pyplot as plt
import easydict

def train(sliced_data, label_name, ):
    data = np.load('./sliced_datas/' + sliced_data + '.npy')#ex.sliced_data = sliced_436
    label = np.load('./labels/'+ label_name+ '.npy')
    X_train = data[:621]
    X_test = data[621:]
    y_train = label[:621]
    y_test = label[621:]
    datadict = {
        'X_train' : X_train,
        'y_train' : y_train,
        'X_test' : X_test,
        'y_test' : y_test
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class mridataset(Dataset):
    def __init__(self, data, data_type):
        if data_type == 'train':
            self.X = data['X_train']
            self.y = data['y_train']
        elif data_type == 'test':
            self.X = data['X_test']
            self.y = data['y_test']
            
        assert len(self.X) == len(self.y), "length should be same between input and label"
        
        self.X = torch.FloatTensor(self.X)#cpu tensor
        self.y = torch.FloatTensor(self.y)#cpu tensor
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return {
            'X': X,
            'y_target': y
        }

    