import os
import time
import pickle
import numpy as np
import librosa

# Load your trained model
with open('speech_command_model.pkl', 'rb') as file:
    import joblib
model = joblib.load('speech_command_model.pkl')


# Function to extract MFCC from audio file
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Initialize alarm variables
alarm_running = False
start_time = 0

while True:
    audio_path = input("Enter WAV filename (or 'exit' to quit): ")

    if audio_path.lower() == 'exit':
        print("Exiting alarm.")
        break

    if not os.path.exists(audio_path):
        print("File not found. Try again.")
        continue

    try:
        # Extract features from the audio file
        features = extract_mfcc(audio_path).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]
        print(f"Predicted Command: {prediction}")

        # Handle alarm based on prediction
        if prediction == 'on':
            if not alarm_running:
                alarm_running = True
                start_time = time.time()
                print("alarm trigerred. Speak 'stop' to end the alarm.")
            else:
                print("alarm already running.")
        elif prediction == 'stop':
            if alarm_running:
                elapsed = time.time() - start_time
                alarm_running = False
                print(f"alarm stopped. you wake up  {round(elapsed, 2)} seconds later.")
            else:
                print("alarm is not running yet.")
        else:
            print("Unknown command. Only 'on' and 'stop' are supported.")
    except Exception as e:
        print("Error processing file:", str(e))
