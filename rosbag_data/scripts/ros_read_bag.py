#!/usr/bin/env python3
import rosbag
import time
import cv2
from cv_bridge import CvBridge


def extract_rgb_from_bag(bag_file):
    bridge = CvBridge()
    count = 1
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera_floor_right/driver/color/image_raw']):  # Adjust topic name as necessary
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'../docker/src/content/spatial_map_data/rgb/rgb_image_{count}.png', cv_image)
                count+=1
            except Exception as e:
                print(e)

if __name__ == '__main__':
    bag_file = '2024-03-20-13-57-17.bag'  # Update this path
    extract_rgb_from_bag(bag_file)