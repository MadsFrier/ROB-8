import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth

def depth_npy2png(data_dir: str, depth_file: str, show: bool) -> None:
    """
    Convert .npy depth file to .png depth file.

    Args:
    data_dir: str: The directory of the depth file.
    depth_file: str: The name of the depth file.
    show: bool: Whether to show the depth image.

    Returns:
    None
    """
    # load depth image that is .npy to array
    depth = load_depth(data_dir + depth_file)
    
    cv2.imwrite(data_dir + depth_file[:-4] + '.png', depth)
        
    if show:
        plt.figure()
        plt.imshow(depth)
        plt.figure()
        plt.imshow(plt.imread(data_dir + depth_file[:-4] + '.png'))
        plt.show()
    return None
    
    
if __name__ == "__main__":
    # choose directory of images to load in
    #data_dir = "/workspaces/ROB-8/docker/src/content/demo_data/depth/" # FOR INSIDE DOCKER
    data_dir = "/home/mads/github/ROB-8/docker/src/content/demo_data/depth/" # FOR OUTSIDE DOCKER
    
    # choose depth image to load in
    depth_name = "5LpN3gDmAk7_1.npy"
    
    # convert depth image to .png and show
    depth_npy2png(data_dir, depth_name, show=True)