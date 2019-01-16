import torch
import numpy as np
import pandas as pd
import codecs
import re
import nltk
import random
from rnn import *
from seed import *

from nltk.stem import WordNetLemmatizer

from random import shuffle

from numpy import array

from sklearn.feature_extraction.text import CountVectorizer

from torch import tensor
from torch import nn
from torch import optim
#from torch.autograd import Variable
import torch.utils.data.dataloader as dataloader

#from scipy.stats import entropy


from scipy.signal import savgol_filter
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from question_dataset import *





# Random seed, don't change it if you don't know what it is
random_seed = 42

use_cuda = torch.cuda.is_available()


def seeding_random():
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic=True
