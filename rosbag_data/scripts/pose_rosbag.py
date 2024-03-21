#!/usr/bin/env python3
#!/usr/bin/env python3
import rosbag
import numpy as np

def save_robot_pose_to_file(position, orientation, count):
    with open(f"../docker/src/content/spatial_map_data/pose/robot_pose_{count}.txt", 'w') as txt_file:
        txt_file.write(f"{position.x}, {position.y}, {position.z}, {orientation.x}, {orientation.y}, {orientation.y}, {orientation.w}")

def extract_robot_positions_from_bag(bag_file):
    count = 1
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/robot_pose']):
            save_robot_pose_to_file(msg.position, msg.orientation, count)
            count += 1
            print(msg)

if __name__ == "__main__":
    bag_file = "2024-03-20-13-57-17.bag"
    extract_robot_positions_from_bag(bag_file)