# Udacity_Python_for_AI
 
This repository contains the Python scripts and the Jupyter notebook necessary to fulfill the requirements of Udacity's 
<a href="https://www.udacity.com/course/ai-programming-python-nanodegree--nd089">AI Programming with Python nanodegree</a>. 

The notebook includes all the instructions and hints provided by Udacity to help you build an image classifier program based on neural network (NN) architecture. It is trained on a collection of images, as described in more detail in the .ipynb file.  

Train.py, predict.py, as well as the required model_funct.py and utility_funct.py serve as command line tools that wrap most of the code found in the notebook into functions to which the NN's hyperparameters (required for fine-tuning) can be passed via user arguments (using the argparse module). Train.py subsequently loads a NN architecture and trains it on a given dataset. Predict.py prints out the top predicted categories and their respective probabilities after receiving and processing a test image. 
 
