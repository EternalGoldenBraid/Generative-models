import torch as t
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os.path as osp
import os

from pathlib import Path

from PIL import Image

class FacesData(Dataset):
    def __init__(self, DEBUG=None, path="faces", limit=1500):
        super().__init__()

        self.x = []
        self.y = None
        self.DEBUG = DEBUG
        self.removed = None
        self.img_list = []
        self.path = path
        self.w = None
        self.h = None

        # Get the list of all files and directories
        self.img_list = os.listdir(path)
        print(f"Loading {limit}/{len(self.img_list)} files.")
        self.img_list = self.img_list[:limit]


    def process(self, size=(64,64), limit=None, shuffle=False, seed=42):
        ### Clean, Downscale and grayscale to single channel.
        self.w, self.h = size

        ## Remove all below treshold size.
        self.removed = []
        cut_treshold = 260
        len1 = len(self.img_list)
        print(f" Images before cleaning: {len1}.")

        for idx, name in enumerate(self.img_list):
            img = Image.open(os.path.join(self.path, name))
            self.x.append(img)
            img = np.array(img)
            if img.shape[0] < cut_treshold or img.shape[1] < cut_treshold: 
                self.x[idx] = None
        self.x = list(filter(None, self.x))

        len2 = len(self.x)
        print(f" Images after cleaning: {len2}, removed: {len1-len2}")

        ## Select a subset
        if limit is not None:
            if shuffle:
                random.seed(seed)
                self.x = list(random.sample(self.x, limit))
            else:
                self.x = self.x[:limit]
        elif shuffle:
            self.x = random.shuffle(self.x)

        ## Resize
        for img in self.x:
            img.thumbnail(size, Image.ANTIALIAS)
            #img.save(outfile, "JPEG") # If doesn't fit in memory

        ## Grayscale
        #self.x = [img.convert('gray')/255 for img in self.x]
        self.x = np.array([np.array(img.convert('L')) for img in self.x])
        self.x = t.tensor(self.x).float()/255
        self.y = np.zeros(len(self.x))
        self.x = self.x.reshape(-1, 1, self.h, self.w)
        print("Processed into shape:", self.x.shape)

    def transform(self, methods=None, n_components=10):

        #x_centered = self.x - np.mean(self.x)
        # PCA
        pca = PCA(n_components)
        self.x = np.array(pca.fit_transform(self.x.T)).T

        if self.DEBUG: print("PCA completed")

    def visualize(self, limit):

        for idx, img in enumerate(self.x):
            if len(np.array(img).shape) > 2: plt.imshow(img)
            else: plt.imshow(img, cmap='gray')
            plt.title(label=np.array(img).shape)
            plt.show()

            if idx > limit: return

        x = np.array(self.x).reshape(-1, len(self.x))
        print(x.shape)

        # SVD
        u, s, v = np.linalg.svd(x)

        fig_svd, axs_svd = plt.subplots(2,1)
        axs_svd[0].plot(s)
        axs_svd[1].plot(np.cumsum(s)/np.sum(s))
        plt.show()

        x_centered = x - np.mean(x)
        C = np.cov(x_centered.T)

        # PCA
        values, vectors = np.linalg.eig(C)

        fig_pca, axs_pca = plt.subplots(2,1)
        axs_pca[0].plot(values)
        axs_pca[1].plot(np.cumsum(values)/np.sum(values))
        plt.show()

        # PCA
        pca = PCA(5)
        pca.fit(C)
        print(pca.components_)

        fig_pca2, axs_pca2 = plt.subplots(2,1)
        axs_pca2[0].plot(pca.singular_values_)
        axs_pca2[1].plot(np.cumsum(pca.singular_values_)/np.sum(pca.singular_values_))
        plt.show()

    def __getitem__(self, idx):
        # No labels
        return self.x[idx], 0
        #return t.tensor(self.x[idx]), t.tensor(self.y[idx])
        pass
    
    def __len__(self):
        return len(self.x)

    def size(self):
        return self.x[0].shape

if __name__=='__main__':
    ds = FacesData()
    #ds.__len__()
    #ds.visualize(limit=3)
    #ds.process(limit=100)
    ds.process()
