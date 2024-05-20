import sys
import os
import time
import threading
from openai import OpenAI
import tkinter as tk
from tkinter import filedialog, scrolledtext
import open3d as o3d
script_dir = os.path.dirname(__file__)
#print(script_dir)
vis_path = '../vis'
lp_path = '../Landmark'
target_vis= os.path.join(script_dir, vis_path)
#print(target_vis)
target_lp = os.path.join(script_dir, lp_path)
sys.path.append(target_vis)
sys.path.append(target_lp)
import landmark_position as lp
#from Landmark import landmark_position as lp
#import display_pose as dp

#vis, pcd, mesh_cylinder, mesh_arrow = dp.vis_init(pcd_path="C:/Users/madsf/Documents/gitHub/ROB-8/gpt/Landmark/rgb_2d_map.pcd")


# Set up OpenAI client
client = OpenAI()
model_version_4o = "gpt-4o"
model_version_3_5_turbo_0125 = "gpt-3.5-turbo-0125"

# Get file and assistant IDs from environment variables
my_file  = client.files.create(
  file=open("gpt\database.txt", "rb"),
  purpose='assistants'
)
assistant = client.beta.assistants.create(
  name="NavBotV2",
  instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant_0, plant_1, chair_0, chair_1, chair_2, package_0, package_1, package_2, package_3 and one additional is robot pose, that you only display when asked by the user, which you need to look in the file of the thread to verify it, when you display the robot pose just write: The robot pose is: and then the pose, so for example: The robot pose is x: 3, y: 2, yaw: 90. Do not add any additional information about where you took the information from, don't reference the database. The only extra scenarios in regards to objects is when the user asks the robot to move to either the close or the furthest object where it can specifiy just the name of the object without the index, so for example the user would ask move to the closest or nearest chair, and then you call the function robot.move_to_closest('chair'), if the user asked for the furthest chair, then,  robot.move_to_furthest('chair') Any other obects is not acceptable and you answer by saying that they are not interactable and only the specified objects in the list are available. The only possible interactions with objects are: inspect, move to, delivery, move in between, move to closest, move to furthest . So every time a user asks for what you can do you specify the available objects and the tasks you can perform.  If the object is in the list and the operation/task is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1') . There are two more types of interactions that do not require objects, but still need the same output format in the sense fo you aagreeing to he user and then do the function call. Those are to move a distance in a certain direction and to rotate an angle in a certain direction. So for example move forward 2 meters, then you should use the function robot.move(distance=2, direction='forward') and the other would be rotate, then if the user asks to rotate 45 degrees to the right, you use robot.rotate(angle=45, direction='right'). The only direction you should output for the move function with the distance specified are: forwaard, back, left and right, as for the rotate function, it only should be only left and right the directions, if the user asks for something that does not fall into these categories you say it to him. The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('',''), robot.move_between('',''), robot.move_to_closest(''), robot.move_to_furthest(''), robot.move(distance='', direction=''), robot.rotate(angle=, direction=''). Don't put quotes around the function please, only for the arguments in the function. Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. So for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1'). If a sequence of valid tasks is called mantain the following type of structure: Yes I can move to chair_1 and then inspect package_0. I am calling the function(s): robot.move_to('chair_1') robot.inspect('package_0')",
  model=model_version_4o,
  tools=[{"type": "file_search"}],
)
# Create a new thread
my_thread = client.beta.threads.create()

class NavBotV2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NavBot V2 GUI")
        
        print('Launching NavBot V2 GUI')

        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        #self.load_button = tk.Button(self.frame, text="Load PCD File", command=self.load_pcd_file)
        #self.load_button.pack(pady=10)

        self.chat_window = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, state='disabled', height=20)
        self.chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_window.tag_configure("bold", font=("Helvetica", 12, "bold"))
        self.chat_window.tag_configure("normal", font=("Helvetica", 12))
        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(padx=10, pady=10, fill=tk.X)

        self.entry = tk.Entry(self.entry_frame, font=("Helvetica", 14))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        self.loading = False  
        self.chat_window.tag_configure("loading", font=("Helvetica", 12, "italic"))
        lp.update_robot_pose(0,0,0)

    
    def loading_animation(self):
        first_run = True
        while self.loading:
            for dots in ["", ".", "..", "..."]:
                if not self.loading:
                    break
                self.chat_window.configure(state='normal')
                if first_run:
                    self.chat_window.insert(tk.END, "\nAssistant is typing", "loading")
                    first_run = False
                else:
                    self.chat_window.delete("loading.first", "loading.last")  # Clear the previous loading text
                    self.chat_window.insert(tk.END, f"\rAssistant is typing{dots}", "loading")
                self.chat_window.configure(state='disabled')
                self.chat_window.update_idletasks()
                time.sleep(0.5)
        self.chat_window.configure(state='normal')
        self.chat_window.delete("loading.first", "loading.last")  # Clear the final loading text
        self.chat_window.configure(state='disabled')



    def send_message(self, event=None):
        user_message = self.entry.get()
        if user_message.strip():
            self.display_message("You: ", "bold" )
            self.display_message(f"{user_message}\n", "normal")
            self.entry.delete(0, tk.END)
            threading.Thread(target=self.get_response, args=(user_message,)).start()

    def display_message(self, message, tag=None):
        self.chat_window.configure(state='normal')
        if tag:
            self.chat_window.insert(tk.END, message, tag)
        else:
            self.chat_window.insert(tk.END, message + "\n")
        self.chat_window.configure(state='disabled')
        self.chat_window.yview(tk.END)

    def get_response(self, user_message):
        # Add user message to the thread
        self.loading = True  # Add this line
        threading.Thread(target=self.loading_animation).start()
        time.sleep(0.25)
        my_thread_message = client.beta.threads.messages.create(
            thread_id=my_thread.id,
            role="user",
            content=user_message,
            attachments=[  # Change v1 to v2: Messages have the attachments parameter instead of the file_ids parameter
                {"file_id": my_file.id, "tools": [{"type": "file_search"}]}
            ],
        )

        # Run the assistant
        my_run = client.beta.threads.runs.create(
            thread_id=my_thread.id,
            assistant_id=assistant.id,
            instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant_0, plant_1, chair_0, chair_1, chair_2, package_0, package_1, package_2, package_3 and one additional is robot pose, that you only display when asked by the user, which you need to look in the file of the thread to verify it, when you display the robot pose just write: The robot pose is: and then the pose, so for example: The robot pose is x: 3, y: 2, yaw: 90. Do not add any additional information about where you took the information from, don't reference the database. The only extra scenarios in regards to objects is when the user asks the robot to move to either the close or the furthest object where it can specifiy just the name of the object without the index, so for example the user would ask move to the closest or nearest chair, and then you call the function robot.move_to_closest('chair'), if the user asked for the furthest chair, then,  robot.move_to_furthest('chair') Any other obects is not acceptable and you answer by saying that they are not interactable and only the specified objects in the list are available. The only possible interactions with objects are: inspect, move to, delivery, move in between, move to closest, move to furthest . So every time a user asks for what you can do you specify the available objects and the tasks you can perform.  If the object is in the list and the operation/task is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1') . There are two more types of interactions that do not require objects, but still need the same output format in the sense fo you aagreeing to he user and then do the function call. Those are to move a distance in a certain direction and to rotate an angle in a certain direction. So for example move forward 2 meters, then you should use the function robot.move(distance=2, direction='forward') and the other would be rotate, then if the user asks to rotate 45 degrees to the right, you use robot.rotate(angle=45, direction='right'). The only direction you should output for the move function with the distance specified are: forwaard, back, left and right, as for the rotate function, it only should be only left and right the directions, if the user asks for something that does not fall into these categories you say it to him. The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('',''), robot.move_between('',''), robot.move_to_closest(''), robot.move_to_furthest(''), robot.move(distance='', direction=''), robot.rotate(angle=, direction=''). Don't put quotes around the function please, only for the arguments in the function. Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. So for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1'). If a sequence of valid tasks is called mantain the following type of structure: Yes I can move to chair_1 and then inspect package_0. I am calling the function(s): robot.move_to('chair_1') robot.inspect('package_0')",
        )

        # Initial delay before the first retrieval
        time.sleep(15)

        # Periodically retrieve the run to check its status
        while my_run.status in ["queued", "in_progress"]:
            keep_retrieving_run = client.beta.threads.runs.retrieve(
                thread_id=my_thread.id, run_id=my_run.id
            )

            if keep_retrieving_run.status == "completed":
                # Retrieve the messages added by the assistant to the thread
                all_messages = client.beta.threads.messages.list(thread_id=my_thread.id)

                # Display assistant message
                assistant_message = all_messages.data[0].content[0].text.value
                print(assistant_message)
                robot_command = lp.extract_function_calls(assistant_message)
                print(f'Robot Command: {robot_command}')
                function_name, argument = lp.get_function_name(robot_command)
                print(f'Function Name: {function_name}, Argument: {argument}')
                landmark_positions = lp.call_function(function_name, argument)
                if landmark_positions == False:
                    break 
                    
                # print(f'GPT Response: {robot_command}')
                #print(assistant_message)
                self.loading = False 
                time.sleep(0.5)
                self.display_message("Assistant: ", "bold")
                time.sleep(0.5)
                self.display_message(f"{assistant_message}\n", "normal")
                
                # Update the robot pose in the visualizer
                #dp.update_robot(vis, pcd, mesh_cylinder, mesh_arrow, landmark_positions[0], landmark_positions[1], landmark_positions[2])
                
                #print('robot pose:', lp.get_robot_pose())
                lp.update_robot_pose(landmark_positions[0], landmark_positions[1], landmark_positions[2])
                print('robot pose:', lp.get_robot_pose())


                break
            elif keep_retrieving_run.status in ["queued", "in_progress"]:
                # Delay before the next retrieval attempt
                time.sleep(2.5)
            else:
                self.loading = False
                break
        
        

if __name__ == "__main__":
    root = tk.Tk()
    gui = NavBotV2GUI(root)
    root.mainloop()