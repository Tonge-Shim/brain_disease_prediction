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