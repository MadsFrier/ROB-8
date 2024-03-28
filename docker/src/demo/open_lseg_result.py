import matplotlib.pyplot as plt
import numpy as np

import cv2

data_dir = '/workspaces/ROB-8/docker/src/content/office_data/'
rgb_folder = 'rgb/'
seg_folder = 'semantic/'
file_name = 'office_1'

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy


rgb_img = plt.imread(data_dir + rgb_folder + file_name + '.jpg')

seg_img = load_npy(data_dir + seg_folder + file_name + '.npy')

plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.subplot(1, 2, 2)
plt.imshow(seg_img)
plt.show()  