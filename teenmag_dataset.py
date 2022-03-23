import torch as t
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os.path as osp

class TeenmagData(Dataset):
    def __init__(self, DEBUG=None, folder="data"):
        super().__init__()

        self.x = None
        self.y = None
        self.val_x = None
        self.DEBUG = DEBUG
        self.colvar = False

        #with open(folder+"/"+"training_x.dat", 'rb') as f:
        with open(osp.join(folder, "training_x.dat"), 'rb') as f:
            data = pickle.load(f)
            self.x = [d[:,:,0] for d in data]
        #with open("training_y.dat", 'rb') as f:
        with open(osp.join(folder, "training_y.dat"), 'rb') as f:
            self.y = pickle.load(f)
        #with open("validation_x.dat", 'rb') as f:
        with open(osp.join(folder, "validation_x.dat"), 'rb') as f:
            data = pickle.load(f)
            self.val_x = [d[:,:,0] for d in data]
    
    def process(self, limit=None, shuffle=False, seed=42):
        # Process raw data
        if limit is not None:
            if shuffle:
                random.seed(seed)
                data = list(zip(self.x, self.y))
                data = random.sample(data, limit)
                self.x, self.y = zip(*data)
                self.x = list(self.x)
                self.y = list(self.y)
            else:
                self.x = self.x[:limit]
                self.y = self.y[:limit]
        elif shuffle:
            data = list(zip(self.x, self.y))
            data = random.shuffle(data)
            self.x, self.y = zip(*data)
            self.x = list(self.x)
            self.y = list(self.y)

        # colvar form. i.e. columns are images.
        #self.x = np.array(self.x).reshape(-1, len(self.x))
        #self.colvar = True
        #self.x = np.array(self.x).reshape(len(self.x),-1)
        #self.y = np.array(self.y).astype(int)
        self.x = t.tensor(np.array(self.x).reshape(len(self.x),-1)).float()
        self.y = np.array(self.y).astype(int)
        self.y = t.tensor(self.y).float()

        print("Processed")

    def transform(self, methods=None, n_components=10):

        #x_centered = self.x - np.mean(self.x)
        # PCA
        pca = PCA(n_components)
        self.x = np.array(pca.fit_transform(self.x.T)).T

        if self.DEBUG: print("PCA completed")

    def visualize(self):


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
        return self.x[idx], self.y[idx]
        #return t.tensor(self.x[idx]), t.tensor(self.y[idx])
        pass
    
    def __len__(self):
        if not self.colvar: return len(self.x)
        else: return self.x.shape[1]
        print("LEN")
        pass

if __name__=='__main__':
    ds = TeenmagData()
    ds.__len__()
    ds.process(limit=100)
    print(ds[1])
