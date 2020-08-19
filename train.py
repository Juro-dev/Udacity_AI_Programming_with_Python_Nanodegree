##### Load packages
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image

import time

from model_funct import model_setup
from model_funct import train_model
from utility_funct import get_input_args1
from utility_funct import process_train_imgs

##### Get input arguments (incl. hyperparameters)
in_arg = get_input_args1()

train_dir = in_arg.data_directory
valid_dir = in_arg.validation_directory
save_dir = in_arg.save_dir
h1 = in_arg.hidden_layer_1
h2 = in_arg.hidden_layer_2
classes = in_arg.classes
epochs = in_arg.epochs
lr = in_arg.learning_rate
dropout = in_arg.dropout
input_model = in_arg.arch
device = in_arg.device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader, validloader, train_data = process_train_imgs(train_dir, valid_dir)

##### Main function for training and saving results
def main():
    
    model, optimizer, criterion = model_setup(input_model, h1, h2, classes, dropout, lr)
    train_model(trainloader, validloader, epochs, device, model, optimizer, criterion)
    ##### Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    torch.save({
        'input_model': input_model,
        'epochs': epochs,
        'h1': h1,
        'h2': h2,
        'dropout': dropout,
        'learning_rate': lr,
        'outputs': classes,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'optimizer.state_dict': optimizer.state_dict(),
        'state_dict': model.state_dict()},
        save_dir + 'model_checkpoint.pth')

main()

