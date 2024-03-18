import numpy as np
import imageio
import matplotlib.pyplot as plt

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
    depth = load_depth(data_dir + 'depth/' + depth_file)
    
    # save depth image as .png
    depth_png = np.array(depth * 255, dtype=np.uint16)
    imageio.imwrite(data_dir + 'depth/' + depth_file[:-4] + '.png', depth_png)
    
    if show:
        plt.imshow(depth_png)
        plt.show()
    return None
    
    
if __name__ == "__main__":
    # choose directory of images to load in
    data_dir = "/workspaces/ROB-8/docker/src/content/demo_data/"
    
    # choose depth image to load in
    depth_name = "5LpN3gDmAk7_130.npy"
    
    # convert depth image to .png and show
    depth_npy2png(data_dir, depth_name, show=True)