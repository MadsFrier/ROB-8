from openai import OpenAI
test = True
def call_ChatGPT(user_prompt):
    completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:personal:master-nav-v2:9NJxyQuj",
    messages=[
        {"role": "system", "content": "ChatGPT for Navigation is a chatbot that generates python scripts from prompts with landmarks."},
        {"role": "user", "content": user_prompt}
    ]
    )
    robot_commands = completion.choices[0].message.content
    print(robot_commands)
    return robot_commands
client = OpenAI()
while True:
    user_prompt =input("Enter your prompt: ")
    answer_gpt = call_ChatGPT(user_prompt)
    if test == True:
        f = open("Single_Object_Test.txt", "a")
        f.write('\nUser Prompt: '+user_prompt+' ')
        f.write('Model Response: ' +answer_gpt)
        f.close()