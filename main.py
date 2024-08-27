# from train import train_model
# from app import run_app

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == "train":
#         # Add your training data paths and labels here
#         file_paths = ["./aryan_audio.mp3", "./ashutosh_audio.mp3", "./nirupam_audio.mp3"]
#         labels = [0, 1, 2]  # Corresponding labels for the audio files
#         train_model(file_paths, labels)
#     else:
#         run_app()

# main.py



# main.py
import sys
import os
import subprocess

# # Check for ffmpeg
# def check_ffmpeg():
#     try:
#         subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print("ffmpeg is installed and accessible.")
#     except FileNotFoundError:
#         print("WARNING: ffmpeg is not found. Please install ffmpeg and add it to your system PATH.")

# print(f"Current working directory: {os.getcwd()}")
# print(f"Python path: {sys.path}")

# check_ffmpeg()

try:
    from config import Config
    print("Successfully imported Config")
except ImportError as e:
    print(f"Error importing Config: {e}")

from train import train_model
from app import run_app

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Add your training data paths and labels here
        file_paths = ["./aryan_audio.mp3", "./ashutosh_audio.mp3", "./nirupam_audio.mp3"]
        labels = [0, 1, 2]  # Corresponding labels for the audio files
        train_model(file_paths, labels)
    else:
        run_app()