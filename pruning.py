#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from functions import *
from learn import *


# # Parameters

# In[2]:


from parameters import *


# In[3]:


print("Start")


# ## Dataloader implementation

# In[4]:



seeding_random()

# Encoding in windows-1252, utf-8 generate error on some char
file = codecs.open("train_all.label", "r+","windows-1252")
data = []
for line in file.readlines():
    data.append(line)
train_data = data[:round(len(data)*devLine)]
dev_data = data[round(len(data)*devLine):]

print("Création training set...")
training_set = QuestionDataset(train_data, nb_input-3)

print("Done!")

print("Création dev set...")
dev_set = QuestionDataset(dev_data, training_set.word_list)
seeding_random()

print("Done!")

print("Création test set...")
file = codecs.open("TREC_test.label", "r+","windows-1252")
test_data = []
for line in file.readlines():
    test_data.append(line)
test_set = QuestionDataset(test_data, training_set.word_list)
seeding_random()

# Création du DataLoader
dataloader_args = dict(shuffle=True, batch_size=nb_batchs, num_workers=1,
                       pin_memory=True, worker_init_fn=seeding_random(), collate_fn=pad_collate)
seeding_random()

train_loader = dataloader.DataLoader(training_set, **dataloader_args)
seeding_random()

dataloader_args_notshuffle = dict(shuffle=False, batch_size=nb_batchs, num_workers=1,
                       pin_memory=True, worker_init_fn=seeding_random(), collate_fn=pad_collate)

dev_loader = dataloader.DataLoader(dev_set, **dataloader_args)
seeding_random()

test_loader = dataloader.DataLoader(test_set, **dataloader_args_notshuffle)
seeding_random()

print("Done!")
print("Création du set de pruning...")
if SIZE_DATA_PRUNING == -1:
    prune_data = data
else:
    prune_data = data[:SIZE_DATA_PRUNING]
prune_set = QuestionDataset(prune_data, training_set.word_list)

prune_loader = dataloader.DataLoader(prune_set, **dataloader_args)
seeding_random()
print("Done!")


print("List of word used:")
print(training_set.word_list)


# # Repartition per classe

# In[5]:


if great_analysis:
    classes = [0,0,0,0,0,0]
    for data, target in train_loader:
        for t in list(target):
            t = t.item()
            classes[t] += 1

    total = sum(classes)
    rep_classes = [c/total*100 for c in classes]
    print("Répartitions des données dans les classes:")
    for i in range(len(rep_classes)):
        print("Classe numéro " + str(i+1) + ": " + str(rep_classes[i]) + "%")


# ## Word occurence repartition

# In[6]:


if great_analysis:

    word_occ = training_set.reparti_word
    word_occ = dict(word_occ)
    
    total = sum([value for key, value in training_set.reparti_word.most_common(len(training_set.reparti_word))])
    
    values = [sum([value for key, value in training_set.reparti_word.most_common(i+1)])/total*100 for i in range(len(training_set.reparti_word))]

    x = np.linspace(0, len(values), len(values))
    fig = plt.figure(figsize=(13, 8)) 
    ax = fig.add_subplot(1,1,1)
    cnn_line, = ax.plot(x, values)

    ax.set(xlabel="Vocabulaire unique", ylabel="Couverture en %")


# 

# In[ ]:





# In[ ]:





# 

# # Pruning implementation

# In[7]:


# Do it after learning
### Will prune neurons
def oracle_prune(rnn, data_loader, nb_times):
    accuracy = -1
    prune_acc = accuracy
    pruned_neur = []
    save_state(rnn, "first_state_rnn")
    save_state(rnn, "pruned_state_rnn")
    pos_prune = -1
    for _ in range(nb_times):
        with torch.enable_grad():
            if accuracy == -1:
                accuracy = getEfficience(rnn, data_loader)*100
            # Compute what will be the best amelioration
            prev_layer = rnn.layers[0]
            min_acc = -1
            for i, l in enumerate(rnn.layers):
                if i != 0:
                    for fromm in range(prev_layer):
                        for to in range(l):
                            get_weights(rnn, i)[to][fromm] = 0
                        acc = getEfficience(rnn, data_loader)*100
                        if min_acc < acc:
                            min_acc = acc
                            pos = {'layer': i, 'nb_prev': fromm}
                            pos_prune = i-1
                        rnn = load_state(rnn, "pruned_state_rnn")
                prev_layer = l
                print("layer "+str(i)+" done! min acc = " + str(min_acc) + " | original acc = "+ str(accuracy))

            # Finally prune
            prune_acc = min_acc
            pruned_neur.append(pos_prune)
            print(pos)
            for to in range(rnn.layers[pos['layer']]):
                get_weights(rnn, pos['layer'])[to][pos['nb_prev']] = 0
        save_state(rnn, "pruned_state_rnn")

    rnn = load_state(rnn, "first_state_rnn")
    ret = rnn.layers
    for prune in pruned_neur:
        print(len)
        ret[prune] -= 1
    return ret, prune_acc
                


# # Using the RNN

# In[8]:


import datetime

seeding_random()

rnn = RNN(nb_inputs = nb_input, layers = LAYERS, nb_outputs=nb_output, learning_rate=lr)
if use_cuda:
    rnn = rnn.to("cuda")

seeding_random()


# In[9]:


begin_time = datetime.datetime.now()

with torch.enable_grad():
    job = learn(rnn, train_loader, dev_loader, nb_epochs, great_analysis)
    
    losses_train = job["losses_train"]
    losses_dev = job["losses_dev"]
    pos_best_rnn = job["pos_best"]
    print("Done :)")
    
end_time = datetime.datetime.now()
print("Learned in " + str(end_time - begin_time))


# In[10]:


# Prune manipulations
accuracy = getEfficience(rnn, prune_loader)*100

pruned_layers, acc_prune = oracle_prune(rnn, prune_loader, 1)

print("after one pass in oracle_pruning, we can remove:" + str(pruned_layers) + " and get an accuracy of " + str(acc_prune))


# ## Error curve

# In[11]:


def update_losses(smooth=1):
    x_train = np.linspace(0, len(losses_train), len(losses_train))
    fig = plt.figure(figsize=(13, 8)) 
    ax_train = fig.add_subplot(1,1,1)
    cnn_line_train, = ax_train.plot(x_train, losses_train)
    cnn_line_train.set_ydata(savgol_filter(losses_train, smooth, 3))
    
    if great_analysis:
        x_dev = np.linspace(0, len(losses_dev), len(losses_dev))
        ax_dev = fig.add_subplot(1,1,1)
        cnn_line_dev, = ax_dev.plot(x_dev, losses_dev)
        cnn_line_dev.set_ydata(savgol_filter(losses_dev, smooth, 3))
    
interact(update_losses, smooth=(5, 500, 2));


# # Analysis on test set

# In[12]:



print("Congratulations!")

rnn.eval()

seeding_random()

correct_train = getEfficience(rnn, train_loader)*100

print("On the training set:")
print("Corrects: " + str(correct_train) + "%")
print()

seeding_random()

correct_dev = getEfficience(rnn, dev_loader)*100

print("On the dev set:")
print("Corrects: " + str(correct_dev) + "%")
print()

seeding_random()

correct_test = getEfficience(rnn, test_loader)*100

mean_entropies = -1

print("On the test set:")
print("Moyenne des entropies: " + str(mean_entropies))
print("Corrects: " + str(correct_test) + "%")

print()

inputs = nb_input
if inputs == -1:
    inputs = len(training_set.word_list)-3

print("A présent, tu peux copier-coller ça dans le doc sur le drive :)")
print(str(inputs)+"\t"+str(lr)+"\t"+str(nb_epochs)+"\t"+str(LAYERS)
      +"\t"+str(nb_batchs)+"\t\t"+str(mean_entropies)+"\t"+str(pos_best_rnn)
      +"\t"+str(correct_train)+"%\t"+str(correct_dev)+"%\t"+str(correct_test)+"%")
print()

