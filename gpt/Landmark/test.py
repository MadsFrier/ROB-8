import json
import re
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

file_path = 'C:/Users/soren/Desktop/Kandidat/P8/ROB8/ROB-8/gpt/ChatGPT/landmarks.json'
with open(file_path, 'r') as file:
    landmark_dict = json.load(file)

def get_angle(landmark, rob_pos):
    angle = np.arctan2(landmark[1] - rob_pos[1], landmark[0] - rob_pos[0])
    return angle

def euler_to_quaternion(angle):
    r = R.from_euler('z', angle)
    quaternion = r.as_quat()
    return quaternion

###########################################################

def move_to(landmark, rob_pos):

    if landmark_dict.get(landmark) is not None:
        landmark_obj = f"{landmark}"
        x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
        #print(f"Landmark: {landmark_obj} at X:{x}  Y:{y}")
        angle = get_angle([x, y], rob_pos)
        quaternion = euler_to_quaternion(angle)
        return x, y, angle
    else:
        print(f"Landmark: {landmark} not found.")
        return False

def inspect(landmark, rob_pos):
        if landmark_dict.get(landmark) is not None:
            landmark_obj = f"{landmark}"
            x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
            angle = get_angle([x, y], rob_pos)
            quaternion = euler_to_quaternion(angle)
            print(f"Landmark: {landmark_obj} at X:{x}  Y:{y}")
            return x, y, angle
        else:
            print(f"Landmark: {landmark} not found.")
            return False

def deliver(landmark1, landmark2, rob_pos):

    if landmark_dict.get(landmark1) is None and landmark_dict.get(landmark2) is None:
        print(f"Landmark: {landmark1} and Landmark: {landmark2} not found.")
        return False
    
    elif landmark_dict.get(landmark1) is None:
        print(f"Landmark: {landmark1} not found.")
        return False
    
    elif landmark_dict.get(landmark2) is None:
        print(f"Landmark: {landmark2} not found.")
        return False
    
    else:
        x1, y1 = landmark_dict[landmark1][0], landmark_dict[landmark1][1]
        x2, y2 = landmark_dict[landmark2][0], landmark_dict[landmark2][1]

        angle1 = get_angle([x1, y1], rob_pos)
        angle2 = get_angle([x2, y2], rob_pos)

        quaternions1 = euler_to_quaternion(angle1)
        quaternions2 = euler_to_quaternion(angle2)

        
        #print(f"Delivering to Landmark: {landmark2} at X:{x2}  Y:{y2} from Landmark: {landmark1} at X:{x1}  Y:{y1}")
        return f"{x1}, {y1}, {angle1}\n{x2}, {y2}, {angle2}"

def move_between(landmark1, landmark2, rob_pos):

    if landmark_dict.get(landmark1) is None and landmark_dict.get(landmark2) is None:
        print(f"Landmark: {landmark1} and Landmark: {landmark2} not found.")
        return False
    
    elif landmark_dict.get(landmark1) is None:
        print(f"Landmark: {landmark1} not found.")
        return False
    
    elif landmark_dict.get(landmark2) is None:
        print(f"Landmark: {landmark2} not found.")
        return False
    
    else:
        x1, y1 = landmark_dict[landmark1][0], landmark_dict[landmark1][1]
        x2, y2 = landmark_dict[landmark2][0], landmark_dict[landmark2][1]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        angle = get_angle([mid_x, mid_y], rob_pos)
        quaternion = euler_to_quaternion(angle)
        print('angle:\n', angle,)

        #print(f"Moving between: {landmark1} at X:{x1}  Y:{y1} and Landmark: {landmark2} at X:{x2}  Y:{y2}. New position {mid_x}, {mid_y}.")
        return mid_x, mid_y, quaternion   
    
def move(distance, rob_pos, direction=None):
    x, y = rob_pos
    yaw = 0  
    
    if direction == "left":
        yaw += math.pi / 2 
    elif direction == "right":
        yaw -= math.pi / 2
    elif direction == "behind":
        yaw += math.pi
    
    new_x = x + distance * math.cos(yaw)
    new_y = y + distance * math.sin(yaw)
    
    return new_x, new_y, yaw

def move_to_closest(landmark, rob_pos):
    min_distance = float('inf')
    closest_landmark = None
    
    for landmark_name, coords in landmark_dict.items():
        if landmark_name.startswith(landmark):
            x2, y2 = coords[0], coords[1]
            x1, y1 = rob_pos[0], rob_pos[1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                min_distance = distance
                closest_landmark = landmark_name
    
    if closest_landmark is None:
        print(f"No {landmark}s found.")
        return False
    
    x, y = landmark_dict[closest_landmark][0], landmark_dict[closest_landmark][1]
    angle = get_angle([x, y], rob_pos)
    quaternion = euler_to_quaternion(angle)

    print(f"The closest {landmark} is {closest_landmark} at coordinates ({x}, {y}).")

    return x, y, angle

def move_to_furthest(landmark, rob_pos):
    max_distance = 0
    furthest_landmark = None
    
    for key, coords in landmark_dict.items():
        if key.startswith(landmark):
            x2, y2 = coords[0], coords[1]
            x1, y1 = rob_pos[0], rob_pos[1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > max_distance:
                max_distance = distance
                furthest_landmark = key
    
    if furthest_landmark is None:
        print(f"No {landmark}s found.")
        return False
    
    x, y = landmark_dict[furthest_landmark][0], landmark_dict[furthest_landmark][1]
    angle = get_angle([x, y], rob_pos)
    quaternion = euler_to_quaternion(angle)


    print(f"The furthest {landmark} is {furthest_landmark} at coordinates ({x}, {y}).")
    return x, y, angle

def rotate(angle, direction, rob_pos):
    angle_radians = np.radians(angle)

    if direction == "left":
        angle_radians *= -1
    elif direction != "right":
        print("Invalid direction. Please specify 'left' or 'right'.")
        return None

    new_angle = angle_radians

    print(f"Rotating {angle} degrees to the {direction}.")
    return rob_pos[0], rob_pos[1], new_angle

print(move_to_furthest('chair', [0,0]))