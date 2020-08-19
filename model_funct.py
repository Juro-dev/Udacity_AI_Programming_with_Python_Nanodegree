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

import time

##### Define the network
def model_setup(input_model, h1, h2, classes, dropout, lr):
    
    inputs = {"vgg11": 25088, "densenet121": 1024, "alexnet": 9216}

    if input_model == 'vgg11':
        model = models.vgg11(pretrained=True)        
    elif input_model == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        input_model = models.alexnet(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(inputs[input_model], h1)),
        ('relu1', nn.ReLU()),
        ('dropout1',nn.Dropout(dropout)),
        ('h1', nn.Linear(h1, h2)),
        ('relu2',nn.ReLU()),
        ('dropout2',nn.Dropout(dropout)),
        ('h2',nn.Linear(h2,classes)),
        ('output', nn.LogSoftmax(dim=1))
                        ]))
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    return model, optimizer, criterion

##### Train, validate, and time the network
def train_model(trainloader, validloader, epochs, device, model, optimizer, criterion):
    model.to(device)

    start_training_time = time.time()

    for epoch in range(epochs):
        
        total_steps = len(trainloader)
        running_loss = 0
        valid_loss = 0
        valid_accuracy = 0
        print_every = 100
        steps = -1
        
        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        
            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():

                    for ii, (inputs2, labels2) in enumerate(validloader):
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        output_val = model.forward(inputs2)
                        valid_loss += criterion(output_val, labels2).item()
                        ps = torch.exp(output_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print("Epoch: {}/{}, Step: {}/{}, Train set loss: {:.3f}, Validation set loss: {:.3f}, Validation accuracy: {:.3f}".format(epoch+1, 
                        epochs, 
                        i+1, 
                        total_steps, 
                        running_loss/print_every, 
                        valid_loss/len(validloader), 
                        valid_accuracy/len(validloader)))
                    
                running_loss = 0
                valid_accuracy = 0

                model.train()

    end_training_time = time.time()

    training_time = end_training_time - start_training_time
    print('\nTraining completed after: {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))