from openai import OpenAI
import json
import re
test = False
file_path = 'ChatGPT/landmarks.json'
with open(file_path, 'r') as file:
    landmark_dict = json.load(file)
def move_to(landmark):
        try:
            landmark_obj = f"{landmark}"
            x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
            return f"Landmark: {landmark_obj} at X:{x}  Y:{y}"
        except KeyError:
            return f"Landmark {landmark} not found."

def inspect(landmark):
        try:
            landmark_obj = f"{landmark}"
            x, y = landmark_dict[landmark_obj][0], landmark_dict[landmark_obj][1]
            return f"Landmark {landmark_obj} at X:{x}  Y:{y}"
        except KeyError:
            return f"Landmark {landmark} not found."

def deliver(landmark1, landmark2):
        try:
           
            try:
                landmark_obj_1 = f"{landmark1}"
                print(landmark_obj_1) 
                x1, y1 = landmark_dict[landmark_obj_1][0], landmark_dict[landmark_obj_1][1]
            except KeyError:
                return f"Landmark {landmark1} not found."
            
            try:
                landmark_obj_2 = f"{landmark2}"
                print(landmark_obj_2)
                x2, y2 = landmark_dict[landmark_obj_2][0], landmark_dict[landmark_obj_2][1]
            except KeyError:
                return f"Landmark {landmark2} not found."
                 
            
            return f"Deliver from {landmark_obj_1} at X:{x1}  Y:{y1} to {landmark_obj_2} at X:{x2}  Y:{y2}"
        except KeyError:
            return f"Try again with other landmark."

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
        return f"Function {function_name} not found."
        

client = OpenAI()
user_prompt =input("Enter your prompt: ")
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:personal::9K3HQeWQ",
  messages=[
    {"role": "system", "content": "ChatGPT for Navigation is a chatbot that generates python scripts from prompts with landmarks."},
    {"role": "user", "content": user_prompt}
  ]
)
if test == True:
    f = open("test_One_obj.txt", "a")
    f.write('\nUser Prompt: '+user_prompt+' ')
    f.write('Model Response: ' +completion.choices[0].message.content)
    f.close()

robot_commands = completion.choices[0].message.content
robot_commands =robot_commands.splitlines()
for command in robot_commands:
    print(f'Command: {command}')
    function_name, argument = get_function_name(command)
    robot_command = call_function(function_name, argument)
    print(f'Response: {robot_command}')
# check if fucntion exist and return with input if not the case for either fucntion or landmark not found