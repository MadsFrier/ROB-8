import rosbag
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageio

def save_camera_depth_to_png(depth_image, count):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        cv_image = np.array(cv_image, dtype=np.uint16)
        #print(cv_image)
        np.save(f'../docker/src/content/spatial_map_data/depth/camera_depth_{count}.npy', cv_image)
        plt.imshow(plt.imread(f'../docker/src/content/spatial_map_data/depth/camera_depth_{count}.npy'))
        plt.show()
    except Exception as e:
        print(e)

def extract_camera_depth_from_bag(bag_file):
    count = 1
    bridge = CvBridge()
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera_floor_right/driver/depth/image_rect_raw']):
            save_camera_depth_to_png(msg, count)
            count += 1
            

if __name__ == "__main__":
    bag_file = "2024-03-20-13-57-17.bag"
    extract_camera_depth_from_bag(bag_file)