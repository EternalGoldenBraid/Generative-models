import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm
from teenmag_dataset import TeenmagData
from faces_dataset import FacesData

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, h, w):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, latent_dims)
        self.linear3 = nn.Linear(hidden_dims, latent_dims)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.N = torch.distributions.Normal(0, 1)
        if device == 'cuda': self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        if device == 'cuda': self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, h, w):
        super(Decoder, self).__init__()
        self.h = h
        self.w = w

        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, input_dims)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, self.h, self.w))

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, h, w):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, hidden_dims, latent_dims,h ,w)
        self.decoder = Decoder(input_dims, hidden_dims, latent_dims, h, w)
        self.h = h
        self.w = w
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
