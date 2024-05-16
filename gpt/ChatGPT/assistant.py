import os
from openai import OpenAI
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Step 1: Create an Assistant
my_assistant = client.beta.assistants.create(
    model="gpt-4o",
    instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant 0, plant 1, chair 0, chair 1, chair 2, package 0, package 1, package 2, package 3. ANy other obects is not acceptable and you asnwer by saying that they are not interactable and only the specified objects in the list are available. THe only possible interactions are inspect, move toa and delivery. SO every time a user asks for what you can do you specify the avialable objects and the tasks you can perform.  If the object is in the list and the operation/tsk is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1') The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('','') Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. SO for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1')",
    name="NavBot",
)
print(f"This is the assistant object: {my_assistant} \n")

# Step 2: Create a Thread
my_thread = client.beta.threads.create()
print(f"This is the thread object: {my_thread} \n")
def call_ChatGPT(user_prompt):
    thread_message = client.beta.threads.messages.create(
    thread_id=my_thread.id,
    role="user",
    content=user_prompt,
    )

# Step 3: Add a Message to a Thread
my_thread_message = client.beta.threads.messages.create(
  thread_id=my_thread.id,
  role="user",
  content="Can you please move to chair_1?",
)
print(f"This is the message object: {my_thread_message} \n")

# Step 4: Run the Assistant
my_run = client.beta.threads.runs.create(
  thread_id=my_thread.id,
  assistant_id=my_assistant.id,
  instructions="You are a navigation bot assistant that  only accepts the following objects to interact with: plant 0, plant 1, chair 0, chair 1, chair 2, package 0, package 1, package 2, package 3. ANy other obects is not acceptable and you asnwer by saying that they are not interactable and only the specified objects in the list are available. THe only possible interactions are inspect, move toa and delivery. SO every time a user asks for what you can do you specify the avialable objects and the tasks you can perform.  If the object is in the list and the operation/tsk is possible to perform the output should be along the lines: Yes I can move to chair_1. I am calling the function(s): robot.move_to('chair_1').  The availbale functions are: robot.move_to(''), robot.inspect(''), robot.deliver('',''). Don't put quotes around the function please, only for the arguments in the function. Where the robot deliver function needs the object from as the first argument, and the object it will deliver to as the second argument. So for example: deliver plant_0 to chair_1, would be robot.deliver('plant_0', 'chair_1')",
)
print(f"This is the run object: {my_run} \n")

# Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
while my_run.status in ["queued", "in_progress"]:
    keep_retrieving_run = client.beta.threads.runs.retrieve(
        thread_id=my_thread.id,
        run_id=my_run.id
    )
    print(f"Run status: {keep_retrieving_run.status}")

    if keep_retrieving_run.status == "completed":
        print("\n")

        # Step 6: Retrieve the Messages added by the Assistant to the Thread
        all_messages = client.beta.threads.messages.list(
            thread_id=my_thread.id
        )

        print("------------------------------------------------------------ \n")

        print(f"User: {my_thread_message.content[0].text.value}")
        print(f"Assistant: {all_messages.data[0].content[0].text.value}")
        break

    elif keep_retrieving_run.status == "queued" or keep_retrieving_run.status == "in_progress":
        pass
    else:
        print(f"Run status: {keep_retrieving_run.status}")
        break