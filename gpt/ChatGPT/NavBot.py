import os
import time
from openai import OpenAI
client = OpenAI()
test = True

assistant_id = 'put_id_here'
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
    )

    # Run the assistant
    my_run = client.beta.threads.runs.create(
        thread_id=my_thread.id,
        assistant_id=assistant_id,
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
            if test == True:
                f = open("Single_Object_Test_Assistant.txt", "a")
                f.write('\nUser Prompt: '+user_input+' ')
                f.write('Assistant Response: ' + assitant_answer)
                f.close()

            break
        elif keep_retrieving_run.status in ["queued", "in_progress"]:
            # Delay before the next retrieval attempt
            time.sleep(2.0)
            pass
        else:
            break