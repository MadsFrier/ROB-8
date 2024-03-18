#!/usr/bin/env python3

import sys
import os
import imageio
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import open3d as o3d

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth


def create_image(data_dir: str, rgb_file: str, depth_file: str, show: bool):

    rgb_dir = os.path.join(data_dir, "rgb/")
    depth_dir = os.path.join(data_dir, "depth/")
    
    rgb_path = rgb_dir + rgb_file
    depth_path = depth_dir + depth_file
        
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    
    if show:
        plt.subplot(1, 2, 1)
        plt.title('RGB')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth')
        plt.imshow(rgbd_image.depth)
        plt.show()    