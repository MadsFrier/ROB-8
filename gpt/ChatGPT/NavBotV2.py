import sys
import os
import time
from openai import OpenAI
script_dir = os.path.dirname(__file__)
print(script_dir)
vis_path = '../vis'
lp_path = '../Landmark'
target_vis= os.path.join(script_dir, vis_path)
print(target_vis)
target_lp = os.path.join(script_dir, lp_path)
sys.path.append(target_vis)
sys.path.append(target_lp)
import landmark_position as lp
#from Landmark import landmark_position as lp
import display_pose as dp
time.sleep(1)
vis, pcd, mesh_cylinder, mesh_arrow = dp.vis_init(pcd_path="C:/Users/Christian/Documents/GitHub/ROB-8/gpt/Landmark/rgb_2d_map.pcd")
time.sleep(5)
client = OpenAI()
test = True

my_file = client.files.create(
  file=open("gpt\database.txt", "rb"),
  purpose='assistants'
)
my_assistant = client.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant_0, plant_1, chair_0, chair_1, chair_2, package_0, package_1, package_2, package_3. ANy other obects is not acceptable and you asnwer by saying that they are not interactable and only the specified objects in the list are available. THe only possible interactions are inspect, move toa and delivery. So every time a user asks for what you can do you specify the avialable objects and the tasks you can perform.  If the object is in the list and the operation/tsk is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1'). Remember to accpet the ma,es with underscores after it, so for example chair_1 is acceptable and so on.  The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('',''). Don't put quotes around the function please, only for the arguments in the function. Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. So for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1'). If a seuqence of valid tasks is called mantain the following type of strucre: Yes I can move to chair_1 and then isnpect package_0. I am calling the function(s): robot.move_to('chair_1') robot.inspect('package_0')",
    name="NavBotV2",
    tools=[{"type": "file_search"}]
)
print(f'This is the assistant id: {my_assistant.id} \n')
# Create a new thread
my_thread = client.beta.threads.create()

# Loop until the user enters "quit"
while True:
    # Get user input
    user_input = input("Enter your input: ")

    # Check if the user wants to quit
    if user_input.lower() == "exit":
        print("\nExiting...")
        break

    # Add user message to the thread
    my_thread_message = client.beta.threads.messages.create(
        thread_id=my_thread.id,
        role="user",
        content=user_input,
        attachments= [
        { "file_id": my_file.id, "tools": [{"type": "file_search"}] }
      ],
    )

    # Run the assistant
    my_run = client.beta.threads.runs.create(
        thread_id=my_thread.id,
        assistant_id=my_assistant.id,
        instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant 0, plant 1, chair 0, chair 1, chair 2, package 0, package 1, package 2, package 3. ANy other obects is not acceptable and you asnwer by saying that they are not interactable and only the specified objects in the list are available. THe only possible interactions are inspect, move toa and delivery. So every time a user asks for what you can do you specify the avialable objects and the tasks you can perform.  If the object is in the list and the operation/tsk is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1').  The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('',''). Don't put quotes around the function please, only for the arguments in the function. Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. So for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1'). If a seuqence of valid tasks is called mantain the following type of strucre: Yes I can move to chair_1 and then isnpect package_0. I am calling the function(s): robot.move_to('chair_1') robot.inspect('package_0')",
    )

    # Initial delay before the first retrieval
    time.sleep(15)

    # Periodically retrieve the run to check its status
    while my_run.status in ["queued", "in_progress"]:
        keep_retrieving_run = client.beta.threads.runs.retrieve(
            thread_id=my_thread.id,
            run_id=my_run.id
        )

        if keep_retrieving_run.status == "completed":
            # Retrieve the messages added by the assistant to the thread
            all_messages = client.beta.threads.messages.list(
                thread_id=my_thread.id
            )

            assitant_answer = all_messages.data[0].content[0].text.value
            # Display assistant message
            print(f"\nAssistant: {all_messages.data[0].content[0].text.value}\n")
            robot_command = lp.extract_function_calls(assitant_answer)
            print(f'Robot Command: {robot_command}')
            function_name, argument = lp.get_function_name(robot_command)
            print(f'Function Name: {function_name}, Argument: {argument}')
            landmark_positions = lp.call_function(function_name, argument)
            if landmark_positions == False:
                exit()
                
            print(f'GPT Response: {robot_command}')
      
            time.sleep(5)
            # Update the robot pose in the visualizer
            dp.update_robot(vis, pcd, mesh_cylinder, mesh_arrow, landmark_positions[0], landmark_positions[1], landmark_positions[2])


            break
        elif keep_retrieving_run.status in ["queued", "in_progress"]:
            # Delay before the next retrieval attempt
            time.sleep(2.0)
            pass
        else:
            break