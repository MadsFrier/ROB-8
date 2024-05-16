import openai
import time

# Set up the API key
openai.api_key = 'your-api-key'

# Create a new thread
thread_response = openai.Thread.create()
thread_id = thread_response['id']
print(f"Thread ID: {thread_id}")

def send_message_and_get_conversation(thread_id, user_message):
    # Send a new message to the thread
    openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        thread_id=thread_id
    )

    # Retrieve all messages in the thread
    thread_messages = openai.Thread.list_messages(thread_id=thread_id)

    # Print the entire conversation
    for message in thread_messages['data']:
        print(f"{message['role']}: {message['content']}")

# Loop to continuously send messages and print conversation history
while True:
    user_message = input("Enter your message: ")
    send_message_and_get_conversation(thread_id, user_message)
    print("\n--- Conversation History ---\n")
    time.sleep(1)  # Optional: Add a delay between iterations if needed
