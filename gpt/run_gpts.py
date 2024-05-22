import subprocess

def run_script(scriptname):
    return subprocess.Popen(["python", scriptname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if __name__ == "__main__":
    # Start both scripts
    process1 = run_script("gpt/ChatGPT/GUI.py")
    process2 = run_script("gpt/ChatGPT/GUI_35.py")