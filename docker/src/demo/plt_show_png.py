import matplotlib.pyplot as plt
import numpy as np
import cv2

path_dir = "/workspaces/ROB-8/docker/src/content/demo_data/"
folder = 'semantic/'
folder1 = 'rgb/'
semantic_name = "5LpN3gDmAk7_140.npy"


semantic_img = np.load(path_dir + folder + semantic_name)
rgb_img = plt.imread(path_dir + folder1 + semantic_name[:-4] + '.png')

plt.subplot(1, 2, 1)
plt.title('RGB')
plt.imshow(rgb_img)
plt.subplot(1, 2, 2)
plt.title('semantic')
plt.imshow(semantic_img)
plt.show()  