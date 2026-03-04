import subprocess

# List of scripts to run in sequence
scripts_to_run = ["preprocess.py", "train_model.py", "quantize_export.py", "inference_server.py"]

for script in scripts_to_run:
    print(f"Running {script}...")
    # The subprocess.run() function waits for the command to complete
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"Error in {script}. Stopping sequence.")
        break
    print(f"Finished {script}\n")
