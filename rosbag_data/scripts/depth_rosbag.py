#!/usr/bin/env python3
import rosbag
import time
import numpy as np
'''
depth_test = np.load('depth_image_1.npy')
print(depth_test)
time.sleep(30)
'''
def extract_rgb_from_bag(bag_file):
    count=1
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera_floor_right/driver/depth/image_rect_raw']):  # Adjust topic name as necessary
            print(msg)
            
            np.save(f'depth_image_{count}', msg)
            count+=1
            time.sleep(30)
          
if __name__ == '__main__':
    bag_file = '2024-03-20-13-57-17.bag'  # Update this path
    extract_rgb_from_bag(bag_file)
