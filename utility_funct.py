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

from PIL import Image

from model_funct import model_setup

##### Function for getting input arguments (incl. hyperparameters) for model setup
def get_input_args1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type = str , default = 'flowers/train', 
                    help = 'path to the training set folder')
    parser.add_argument('--validation_directory', type = str , default = 'flowers/valid', 
                    help = 'path to the validation set folder')
    parser.add_argument('--save_dir', type = str , default = 'chckpt_save/', 
                    help = 'folder for saving trained model checkpoint')
    parser.add_argument('--arch', type = str , default = 'vgg11', choices=['vgg11', 'densenet121', 'alexnet'],
                    help = 'choice of neural network model and corresponding # of input nodes')
    parser.add_argument('--hidden_layer_1', type = int , default = 1024, 
                    help = '# of nodes in h1')
    parser.add_argument('--hidden_layer_2', type = int , default = 512, 
                    help = '# of nodes in h2')

    parser.add_argument('--classes', type = int , default = 102, 
                    help = '# of classes you are trying to predict')

    parser.add_argument('--epochs', type = int , default = 25, 
                    help = '# of epochs for training the network')
    parser.add_argument('--learning_rate', type = int , default = 0.001, 
                    help = 'learning rate')
    parser.add_argument('--dropout', type = int , default = 0.15, 
                    help = '# of nodes in h1')
    parser.add_argument('--device', type = str , default = 'cuda:0', choices=['cuda:0', 'cpu'],
                    help = 'device for training (cpu or gpu')

    return parser.parse_args()

##### Function for getting input arguments for loading checkpoint and mapping
def get_input_args2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str , default = 'flowers/test/33/image_06460.jpg', 
                    help = 'path to the test image')
    parser.add_argument('--checkpoint', type = str , default = 'chckpt_save/model_checkpoint.pth', 
                    help = 'path to the model checkpoint')
    parser.add_argument('--k_most_likely', type = int , default = 5, 
                    help = 'k likeliest classes')
    parser.add_argument('--category_names', type = str , default = 'cat_to_name.json',
                    help = 'dictionary of your numerically-indexed classes')
    parser.add_argument('--device', type = str , default = 'cuda:0', choices=['cuda:0', 'cpu'],
                    help = 'device for testing (gpu or cpu')

    return parser.parse_args()

##### Function for processing your training and validation images
def process_train_imgs(train_dir, valid_dir):
    ##### Define the transformations of the original images
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    ##### Load data and apply transformations
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

    ##### Define trainloader to operationalize data for training
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)


    return trainloader, validloader, train_data

##### Function for scaling, cropping, and normalizing a PIL image for a PyTorch model, returns an Numpy array
def process_image(image):
    pil_img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    img_transform = img_transforms(pil_img)
    
    return img_transform

##### Function for loading the saved model from checkpoint
def load_checkpoint(file):
    checkpoint = torch.load(file)
    input_model = checkpoint['input_model']
    h1 = checkpoint['h1']
    h2 = checkpoint['h2']
    classes = checkpoint['outputs']
    lr = checkpoint['learning_rate']
    dropout = checkpoint['dropout']
    model,_,_ = model_setup(input_model, h1, h2, classes, dropout, lr)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

##### Function for predicting the class (or classes) of an image using a trained deep learning model

def predict(image_path, model, device, topk=5):
    img_new = process_image(image_path)
    img_new = img_new.unsqueeze_(0)
    model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        img_new = img_new.to(device)
        outputs = model.forward(img_new)
        
    ps = F.softmax(outputs.data, dim=1)
    top_p, top_class = ps.topk(topk, dim=1)
    
    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
    
    classes = []
    cpu_labels = top_class.cpu()
    for label in cpu_labels.detach().numpy()[0]:
        classes.append(class_idx_dict[label])
        
    return top_p.cpu().numpy()[0],classes

##### Function for displaying an image along with the top 5 classes
def plot_prediction(image, cat_to_name, flower_labels, final_class, probs):

    pil_img = Image.open(image)

    fig, ax = plt.subplots(figsize=(10,10), nrows=2, ncols=1, squeeze=False)
    fig.suptitle('Sanity checking')

    ax[0,0].imshow(pil_img)
    ax[0,0].axis('off')
    ax[0,0].set_title(cat_to_name[final_class])

    y_ticks = np.arange(len(flower_labels))

    ax[1,0].barh(y_ticks, probs)
    ax[1,0].set_yticks(y_ticks)
    ax[1,0].set_yticklabels(flower_labels)
    ax[1,0].invert_yaxis()

    plt.show()
