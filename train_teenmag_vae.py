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

from variational_autoencoder import VariationalAutoencoder as VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def plot_img():
    # Display image and label.
    train_features, train_labels = next(iter(data))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0,0,:, :]
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

#ds = FacesData(limit=4590)
ds = FacesData(limit=450)
ds.process(limit=100000)
h, w = ds.size()
ds.x = ds.x.reshape(-1, 1, h, w)

latent_dims = 2
input_dims = h*w
hidden_dims = 2048
batch_size = 1024
epochs = 2000
data = DataLoader(ds, batch_size=batch_size, shuffle=True)


print(f"(Training on {device}.")
vae = VAE(input_dims, hidden_dims, latent_dims, h, w).to(device) # GPU
vae = train(vae, data, epochs)
torch.save(vae,'dumps/vae_dump')


