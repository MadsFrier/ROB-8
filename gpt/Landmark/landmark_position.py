from openai import OpenAI
import json
import re
import open3d as o3d
import numpy as np

file_path = 'ChatGPT/landmarks.json'
with open(file_path, 'r') as file:
    landmark_dict = json.load(file)

def move_to(landmark):

    if landmark_dict.get(landmark) is not None:
        landmark_obj = f"{landmark}"
        x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]

        print(f"Landmark: {landmark_obj} at X:{x}  Y:{y}")
        return x, y
    else:
        print(f"Landmark: {landmark} not found.")
        return False

def inspect(landmark):
        if landmark_dict.get(landmark) is not None:
            landmark_obj = f"{landmark}"
            x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
            print(f"Landmark: {landmark_obj} at X:{x}  Y:{y}")
            return x, y
        else:
            print(f"Landmark: {landmark} not found.")
            return False

def deliver(landmark1, landmark2):

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
        print(f"Delivering to Landmark: {landmark2} at X:{x2}  Y:{y2} from Landmark: {landmark1} at X:{x1}  Y:{y1}")
        return x1, y1, x2, y2
    
def get_function_name(string):
    match = re.match(r"(.+)\('(.+)'\)", string)
    if match:
        function_name = match.group(1)
        argument = match.group(2)
        return function_name, argument
    else:
        return None

def call_function(function_name, argument):
    if function_name == 'robot.move_to':
        return move_to(argument)
    elif function_name == 'robot.inspect':
        return inspect(argument)
    elif function_name == 'robot.deliver':
        landmarks = argument.split(',')
        return deliver(landmarks[0].replace("'", ""), landmarks[1].replace("'", "").strip())
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

def extract_function_calls(text):
    # Define the keyword to look for
    keyword = "I am calling the function(s): "
    
    # Find the starting index of the keyword
    start_index = text.find(keyword)
    
    # If the keyword is found, extract the rest of the text
    if start_index != -1:
        # Add the length of the keyword to the index to start extracting after it
        return text[start_index + len(keyword):].strip()
    else:
        # Return None if the keyword is not found
        return None

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
        