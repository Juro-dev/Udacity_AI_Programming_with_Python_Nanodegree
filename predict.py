##### Load packages
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
from collections import OrderedDict

from PIL import Image

from utility_funct import get_input_args2
from utility_funct import load_checkpoint
from utility_funct import process_image
from utility_funct import predict
# from utility_funct import plot_prediction

##### Get input arguments
in_arg2 = get_input_args2()
device = in_arg2.device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(in_arg2.category_names, 'r') as f:
    cat_to_name = json.load(f)

##### Suppress scientific notation for numpy arrays   
np.set_printoptions(formatter={'float': '{:.12f}'.format}, suppress=True)


##### Main function for making and printing prediction
def main2():
    model_load = load_checkpoint(in_arg2.checkpoint)
    probs, classes = predict(in_arg2.image_path, model_load, device, in_arg2.k_most_likely)
    flower_labels = [cat_to_name[i] for i in classes]
    top_result = np.argmax(probs)
    final_class = classes[top_result]

    print("Most probable class of image: {}, Probability: {:.1f}%, k-most probable classes of image: {}, k-highest probabilities: {}\n".format(cat_to_name[final_class], 
                        probs[top_result] * 100, 
                        flower_labels, 
                        probs))

    # plot_prediction(in_arg2.image_path, cat_to_name, flower_labels, final_class, probs)


main2()
