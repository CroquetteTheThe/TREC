import torch
import numpy as np
from rnn import *
from seed import *

from torch import tensor
from torch import nn
from torch import optim


# RNN implementation
# Using ReLU, and CrossEntropy

class RNN(nn.Module):
    def __init__(self, nb_inputs, layers, nb_outputs, learning_rate):
        super(RNN, self).__init__()
        
        # Applying RNN layer, and softmax then
        prev_layer = nb_inputs
        for i, l in enumerate(layers):
            name_attr = "rnn"+str(i)
            setattr(self, name_attr, nn.RNN(input_size=prev_layer, num_layers=1,
               hidden_size=l, dropout=0., batch_first=True, nonlinearity='relu'))
            prev_layer = l
        
        name_attr = "rnn"+str(len(layers))
        setattr(self, name_attr, nn.RNN(input_size=prev_layer, num_layers=1,
           hidden_size=nb_outputs, dropout=0., batch_first=True, nonlinearity='relu'))
        
        
        self.sm = nn.Softmax(dim=1)
        
        # Other usefull variables here
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.input_dim = nb_inputs
        self.output_dim = nb_outputs
        self.layers = layers
        #print(self.__dict__)

    def forward(self, inputs):
        inp = inputs
        
        for i in range(len(self.layers)):
            h0 = torch.zeros(1, inp.size(0), self.layers[i])
            if use_cuda:
                h0 = h0.to("cuda")
            name_attr = "rnn"+str(i)
            #print(getattr(self, name_attr))
            inp, hn = getattr(self, name_attr)(inp, h0)
         
        h0 = torch.zeros(1, inp.size(0), self.output_dim)
        if use_cuda:
            h0 = h0.to("cuda")
        name_attr = "rnn"+str(len(self.layers))
        #print(getattr(self, name_attr))
        inp, hn = getattr(self, name_attr)(inp, h0)

        x = self.sm(hn[0])
        return x

# End of the class RNN


# return correct_percent
def getEfficience(rnn, batch_list) :
    total_correct = 0
    total = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    for (data, target) in batch_list :
        data, target = data.to(device), target.to(device)
        out = rnn(data).data
        
        _, predicted = torch.max(out.data, dim=1)
        total_correct += (predicted == target).sum().item()
        total += target.size(0)

    return total_correct / total
