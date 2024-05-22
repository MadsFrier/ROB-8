from openai import OpenAI
import json
import re
import open3d as o3d
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

file_path = 'gpt/ChatGPT/landmarks.json'
with open(file_path, 'r') as file:
    landmark_dict = json.load(file)

def get_angle(landmark, rob_pos):
    angle = np.arctan2(landmark[1] - rob_pos[1], landmark[0] - rob_pos[0])
    return angle

def euler_to_quaternion(angle):
    r = R.from_euler('z', angle)
    quaternion = r.as_quat()
    return quaternion

##################################################

def get_robot_pose():
    with open('gpt/ChatGPT/robot_pose.json', 'r') as file:
        robot_pose = json.load(file)
        x, y, yaw = robot_pose["robot_pose"]
    
    return [x, y, yaw]

def update_robot_pose(x, y, yaw):
    with open('gpt/ChatGPT/robot_pose.json', 'r') as file:
        robot_pose = json.load(file)
        robot_pose["robot_pose"] = [x, y, yaw]
    
    with open('gpt/ChatGPT/robot_pose.json', 'w') as file:
        json.dump(robot_pose, file, indent=4)

def move_to(landmark, rob_pos):

    if landmark_dict.get(landmark) is not None:
        landmark_obj = f"{landmark}"
        x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
        print(f"Landmark: {landmark_obj} at X:{x}  Y:{y}")
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

        
        print(f"Delivering to Landmark: {landmark2} at X:{x2}  Y:{y2} from Landmark: {landmark1} at X:{x1}  Y:{y1}")
        return x1, y1, angle1, x2, y2, angle2
    
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
        #print('angle:\n', angle,)

        #print(f"Moving between: {landmark1} at X:{x1}  Y:{y1} and Landmark: {landmark2} at X:{x2}  Y:{y2}. New position {mid_x}, {mid_y}.")
        return mid_x, mid_y, angle

def move(distance, rob_pos, direction=None):
    x, y = rob_pos
    yaw = 0  
    
    if direction == "left":
        yaw += math.pi / 2 
    elif direction == "right":
        yaw -= math.pi / 2
    elif direction == "behind" | direction == "back" | direction == "backward":
        yaw += math.pi
    elif direction == "forward":
        pass
    else:
        print('Invalid direction. Please specify "left", "right", "foward" "behind".')
        return None
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
    

    if direction == "left":
        pass
    elif direction == "right":
        angle *= -1
    else:
        print("Invalid direction. Please specify 'left' or 'right'.")
        return None
        
    

    print(f"rotating {angle} degrees to the {direction}.")
    return rob_pos[0], rob_pos[1], angle

##################################################

def get_function_name(string):
    match = re.match(r"(.+)\('(.+)'\)", string)
    call_match = re.match(r"(\w+\.\w+)\((.*)\)", string)
    if match:
        function_name = match.group(1)
        argument = match.group(2)
        return function_name, argument

    elif call_match:
        function_call = call_match.group(1)
        arguments_string = call_match.group(2)

        # Split arguments by ',' not within parentheses or quotes
        arguments = re.findall(r"\w+\s*=\s*[^,]+", arguments_string)
        arguments = [arg.strip() for arg in arguments]  # Remove any surrounding whitespace
        print(f'Function Call: {function_call}, Arguments: {arguments}')
        return  function_call, arguments

    else:
        return None

def call_function(function_name, argument):
    if function_name == 'robot.get_robot_pose':
        return get_robot_pose()
    elif function_name == 'robot.move_to':
        return move_to(argument, get_robot_pose()[0:-1])
    elif function_name == 'robot.inspect':
        return inspect(argument, get_robot_pose()[0:-1])
    elif function_name == 'robot.deliver':
        landmarks = argument.split(',')
        return deliver(landmarks[0].replace("'", ""), landmarks[1].replace("'", "").strip(), get_robot_pose()[0:-1])
    elif function_name == 'robot.move_between':
        landmarks = argument.split(',')
        return move_between(landmarks[0].replace("'", ""), landmarks[1].replace("'", "").strip(), get_robot_pose()[0:-1])
    elif function_name == 'robot.move':
        distance  = re.search(r"distance\s*=\s*(\d+)", argument[0])
        distance = float(distance.group(1))
        print('distance:', distance)
        direction = re.search(r"direction\s*=\s*'([^']*)'", argument[1])
        direction = direction.group(1)
        print('direction:', direction)
        return move(distance, get_robot_pose()[0:-1], direction=direction)
    elif function_name == 'robot.move_to_closest':
        args = argument.split(',')
        print('args:', args)
        landmark = args[0].strip()
        print('landmark:', landmark)
        return move_to_closest(landmark, get_robot_pose()[0:-1])
    elif function_name == 'robot.move_to_furthest':
        args = argument.split(',')
        landmark = args[0].strip()
        return move_to_furthest(landmark, get_robot_pose()[0:-1])
    elif function_name == 'robot.rotate':
        angle  = re.search(r"angle\s*=\s*(\d+)", argument[0])
        angle = float(angle.group(1))
        print('angle:', angle)
        direction = re.search(r"direction\s*=\s*'([^']*)'", argument[1])
        direction = direction.group(1)
        print('direction:', direction)
        return rotate(angle, direction, get_robot_pose()[0:-1])
    else:
        f"Function {function_name} not found."
        return False
        
def call_ChatGPT(user_prompt):
    completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:personal:master-nav-v2:9NJxyQuj",
    messages=[
        {"role": "system", "content": "ChatGPT for Navigation is a chatbot that generates python scripts from prompts with landmarks."},
        {"role": "user", "content": user_prompt}
    ]
    )
    robot_commands = completion.choices[0].message.content
    return robot_commands

def landmark_pc(landmark_positions):
    pcd_path = "C:/Users/Christian/Documents/University/ROB8/ROB8 - Code/Landmark/rgb_2d_map.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    for i in range(0, len(landmark_positions), 2):
        x1, y1, z1 = landmark_positions[i], landmark_positions[i+1], 0.001
        landmark_list = [[x1, y1, z1], [x1, y1+0.03, z1], [x1+0.03, y1, z1], [x1-0.03, y1, z1], [x1, y1-0.03, z1], [x1+0.03, y1+0.03, z1], [x1-0.03, y1-0.03, z1], [x1+0.03, y1-0.03, z1], [x1-0.03, y1+0.03, z1]]
        pcd_points, pcd_colours = np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).astype(np.uint8)
        for i in range(len(landmark_list)):
            pcd_points = np.vstack((pcd_points, landmark_list[i]))
            pcd_colours = np.vstack((pcd_colours, [255, 0, 0]))
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours/255.0)
        o3d.visualization.draw_geometries([pcd], zoom=0.27999999999999958,
                                front=[ 0.081165611950153704, 0.022528435461310507, 0.99644599102631892 ],
                                lookat=[ -0.5374871461257219, 2.4917603057239077, -0.0075029322434033048 ],
                                up=[ 0.78370178016044789, 0.6162495248044797, -0.077769164529381679 ])



    current_pose = (0, 0, 0, 0, 0, 0, 1)
    
    x, y, z, qx, qy, qz, qw = current_pose
    
    yaw = math.atan2(2*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
    
    if direction == "left":
        yaw += math.pi/2 
    elif direction == "right":
        yaw -= math.pi/2
    elif direction == "behind":
        yaw += math.pi
    
    if direction is None or direction == "forward":
        new_x = x + distance * math.cos(yaw)
        new_y = y + distance * math.sin(yaw)
    else:
        new_x = x
        new_y = y
    
    return new_x, new_y, z, qx, qy, qz, qw

def extract_all_functions(script):
    # This regex will match function calls and capture the function name and arguments
    matches = re.findall(r"(\w+\.\w+)\s*\((.*?)\)", script)
    
    # Create a list of strings in the format "function_name(arguments)"
    function_calls = [f"{func}({args})" for func, args in matches]
    return function_calls

def extract_function_calls(text):
    # Define the keyword to look for
    keyword = "robot."
    
    # Find the starting index of the keyword
    start_index = text.find(keyword) - 6
    
    # If the keyword is found, extract the rest of the text
    if start_index != -1:
        # Add the length of the keyword to the index to start extracting after it
        return text[start_index + len(keyword):].strip()
    else:
        # Return None if the keyword is not found
        return None
def main():
    client = OpenAI()
    user_prompt =input("Enter your prompt: ")
    robot_commands = None

    if user_prompt != '0':
        robot_commands =  call_ChatGPT(user_prompt)
        robot_commands = extract_function_calls(robot_commands)
    else:
        robot_commands =  call_ChatGPT("move to chair_0")
        robot_commands = extract_function_calls(robot_commands)

    robot_commands =robot_commands.splitlines()
    new_prompt = False
    while True:
    
        for command in robot_commands:
            print(f'Command: {command}')
            function_name, argument= get_function_name(command)
            landmark_positions = call_function(function_name, argument)
        
            if landmark_positions == False:
                user_prompt = input('Please enter a valid command: ')
                robot_commands = call_ChatGPT(user_prompt)
                robot_commands = extract_function_calls(robot_commands).splitlines()
                print(f'Prompt: {robot_commands}')
                new_prompt = True
                continue
            
            else: 
                new_prompt = False
                landmark_pc(landmark_positions)

        if new_prompt == True:
            continue
        
        else:
            exit_input = input('Would you like to exit? (y/n)')
            if exit_input == 'y':
                print('Exiting...')
                exit()
            else:
                robot_commands = call_ChatGPT(input('Enter a new command: '))
                robot_commands = extract_function_calls(robot_commands).splitlines()
            