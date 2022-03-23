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
device

t.load('dumps/vae_dump')

#data = torch.utils.data.DataLoader(
#        #torchvision.datasets.MNIST('./data', 
#        torchvision.datasets.FashionMNIST('./data', 
#               transform=torchvision.transforms.ToTensor(), 
#               download=True),
#        batch_size=128,
#        shuffle=True)
#h, w = 28, 28

#ds = TeenmagData()
#h, w = ds.x[0].shape[0], ds.x[0].shape[1]
#ds.process(limit=100000, shuffle=True)
#ds.x = ds.x.reshape(-1, 1, h, w)

ds = FacesData(limit=4590)
ds.process(limit=100000)
h, w = ds.size()
ds.x = ds.x.reshape(-1, 1, h, w)

latent_dims = 2
input_dims = h*w
hidden_dims = 2048
batch_size = 1024
epochs = 2000
data = DataLoader(ds, batch_size=batch_size, shuffle=True)

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

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in tqdm(enumerate(data), total=len(data)):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w_ = w
    img = np.zeros((n*h, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(h, w).to('cpu').detach().numpy()
            #print(x_hat.shape)
            img[(n-1-i)*w_:(n-1-i+1)*w_, j*w_:(j+1)*w_] = x_hat

    plt.imshow(img, extent=[*r0, *r1], cmap='gray')

plot_latent(autoencoder, data)
plot_reconstructed(autoencoder)
plot_latent(vae, data)
plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))


def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(h, w)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

x, y = data.__iter__().next() # hack to grab a batch
x_1 = x[y == 0][1].to(device) # find a 1
x_2 = x[y == 0][-1].to(device) # find a 0


# In[91]:


interpolate(vae, x_1, x_2, n=20)


# In[92]:


interpolate(autoencoder, x_1, x_2, n=20)


# I also wanted to write some code to generate a GIF of the transition, instead of just a row of images. The code below modifies the code above to produce a GIF.

# In[57]:


from PIL import Image

def interpolate_gif(autoencoder, filename, x_1, x_2, n=10):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255
    
    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning
    
    images_list[0].save(
        f'{filename}.gif', 
        save_all=True, 
        append_images=images_list[1:],
        loop=1)
    
    return images_list


# In[58]:


imgs = interpolate_gif(vae, "vae", x_1, x_2)
img = Image.open('vae.gif').convert('RGB')
plt.imshow(img)
plt.show()
