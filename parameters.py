### Parameters ###

import torch

# Number of recognized words you put in input
nb_input = 1700 # write -1 if you want every words

# Number of classe, constant
nb_output = 6

# Number of hidden layers
LAYERS = [8,8,8,8,8, 8,7,8,8,8]

# Learning rate
lr = 0.001

# Number of epochs
nb_epochs = 80

nb_batchs = 16

pruning_max_drop_rate = 1

# How many percent of your data do you use as training set
devLine = 0.7

SIZE_DATA_PRUNING = -1 #-1 if you want all the data from train + dev set

use_cuda = torch.cuda.is_available()

# If your goal is to draw graphs
great_analysis = False
