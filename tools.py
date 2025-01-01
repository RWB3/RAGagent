 # tools.py

import subprocess

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"