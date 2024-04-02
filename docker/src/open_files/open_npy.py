import matplotlib.pyplot as plt
import numpy as np
import cv2

data_dir = '/workspaces/ROB-8/docker/src/content/rs_data/'
data_folder = 'depth/'
file_name = 'rs_0.npy'

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy


img = load_npy(data_dir + data_folder + file_name)

plt.imshow(img)
plt.show()