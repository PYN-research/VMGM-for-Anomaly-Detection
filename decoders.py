import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
# Decoders
from utils1 import make_nn


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(256, 256)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        #self.net = make_nn(input_size, output_size, hidden_sizes)
        self.net = nn.Linear(input_size, output_size)
    def __call__(self, x):
        pass

class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return torch.distributions.Bernoulli(logits=mapped)
    

class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
    def __call__(self, x):
        #print('x_shape:',x.shape)#[256, 64]
        mean = self.net(x)
        #var = torch.ones_like(mean)
        #print('mean_shape:',mean.shape)#[64, 48, 35]
        #print('var_shape:',var.shape)#[64, 48, 35]
        return mean #torch.distributions.Normal(mean, var)  