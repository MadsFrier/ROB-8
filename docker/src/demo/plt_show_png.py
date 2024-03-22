import matplotlib.pyplot as plt
import numpy as np
import cv2

# path_dir = "/workspaces/ROB-8/docker/src/content/rs_data/" FOR INSIDE DOCKER
path_dir = "/home/mads/github/ROB-8/docker/src/content/rs_data/"
sem_folder = 'semantic/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'
file_name = "rs_0"

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

#semantic_img = np.load(path_dir + sem_folder + file_name + '.npy')
rgb_img = plt.imread(path_dir + rgb_folder + file_name + '.png')
depth_img = load_npy(path_dir + depth_folder + file_name + '.npy')


plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.subplot(1, 2, 2)
plt.imshow(depth_img)
plt.show()  