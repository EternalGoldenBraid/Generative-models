from faces_dataset import FacesData
import numpy as np
import matplotlib.pyplot as plt

ds = FacesData()
img = np.array(ds.x[81])
print(img.shape[0], img.shape[1])
print(img.shape[0] < 300)
plt.imshow(img)
plt.title(img.shape)
plt.show()

removed = ds.x.pop(81)
img = np.array(ds.x[81])
plt.imshow(img)
plt.title(img.shape)
plt.show()

img = np.array(removed)
plt.imshow(img)
plt.title(f"Removed: {img.shape}")
plt.show()

len1 = len(ds)
print(f" Images before cleaning: {len1}.")

removed = []
cut_treshold = 260
for idx, img in enumerate(ds.x):
    if np.array(img).shape[0] < cut_treshold: removed.append(ds.x.pop(idx))

len2 = len(ds)
print(f" Images after cleaning: {len2}.")

#ds.visualize(limit=100)

for idx, img in enumerate(removed):
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"{img.shape}, {idx}/{len1-len2}")
    plt.show()
