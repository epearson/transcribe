import argparse
import json
import os
from dotenv import load_dotenv
from os.path import join, dirname
import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
import whisper

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

whisperModel = whisper.load_model("base.en")
# print("Whisper model loaded.")

def transcribe(file):
    result = whisperModel.transcribe(file, fp16=False)
    return result

def doTranscribe(file_path):
    if not os.path.exists(file_path):
        print("{error: File not found.}")
        return
    
    transcription = transcribe(file_path)
    print(json.dumps(transcription))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio file and output as JSON.')
    parser.add_argument('file_path', type=str, help='Path to the audio file')
    args = parser.parse_args()

    doTranscribe(args.file_path)
