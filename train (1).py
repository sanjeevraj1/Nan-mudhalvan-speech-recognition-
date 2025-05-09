import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Path to your dataset folder
dataset_path = 'dataset'

# Function to extract MFCC features
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Using 13 MFCC features
    return np.mean(mfcc.T, axis=0)

# Loop through the dataset folder and extract features
features = []
labels = []
for label in ['on', 'stop']:  # Assuming you have these labels in your dataset folder
    label_folder = os.path.join(dataset_path, label)
    for file in os.listdir(label_folder):
        if file.endswith('.wav'):  # Assuming audio files are in WAV format
            audio_path = os.path.join(label_folder, file)
            mfcc_features = extract_mfcc(audio_path)
            features.append(mfcc_features)
            labels.append(label)

# Convert features and labels into numpy arrays
features = np.array(features)
labels = np.array(labels)

# Show the shape of features and labels to verify
print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")

# Visualize the distribution of 'on' and 'stop' commands
# Visualize the distribution of 'on' and 'stop' commands
on_count = np.sum(labels == 'on')  # Count of 'on' labels
stop_count = np.sum(labels == 'stop')  # Count of 'stop' labels

plt.figure(figsize=(6, 4))
plt.bar(['on', 'Stop'], [on_count, stop_count], color=['blue', 'orange'], edgecolor='black')
plt.title("Distribution of 'on' and 'Stop' Commands")
plt.xlabel("Commands")
plt.ylabel("Frequency")
plt.show()


import librosa.display

# Choose one "on" and one "stop" sample (you can choose any sample)
on_sample_idx = np.where(labels == 'on')[0][0]  # First 'on' sample
stop_sample_idx = np.where(labels == 'stop')[0][0]  # First 'stop' sample

# Load the corresponding audio files to plot the MFCC
on_audio_path = os.path.join(dataset_path, 'on', os.listdir(os.path.join(dataset_path, 'on'))[on_sample_idx])
stop_audio_path = os.path.join(dataset_path, 'stop', os.listdir(os.path.join(dataset_path, 'stop'))[stop_sample_idx])

on_audio, sr = librosa.load(on_audio_path, sr=None)
stop_audio, sr = librosa.load(stop_audio_path, sr=None)

# Compute MFCC for both samples
on_mfcc = librosa.feature.mfcc(y=on_audio, sr=sr, n_mfcc=13)
stop_mfcc = librosa.feature.mfcc(y=stop_audio, sr=sr, n_mfcc=13)

# Plot MFCC of 'on' command
plt.figure(figsize=(12, 4))
librosa.display.specshow(on_mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2.0f dB")
plt.title('MFCC of "on" Command')
plt.show()

# Plot MFCC of 'stop' command
plt.figure(figsize=(12, 4))
librosa.display.specshow(stop_mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2.0f dB")
plt.title('MFCC of "Stop" Command')
plt.show()
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create and train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set

# Make predictions
predictions = classifier.predict(features)

# Calculate accuracy
accuracy = accuracy_score(labels, predictions) * 100  # Convert to percentage

# Generate classification report
report = classification_report(labels, predictions, target_names=['on', 'stop'], output_dict=True)

# Extract precision, recall, and F1-score for both classes (on and stop)
on_precision = report['on']['precision'] * 100
stop_precision = report['stop']['precision'] * 100
on_recall = report['on']['recall'] * 100
stop_recall = report['stop']['recall'] * 100
on_f1 = report['on']['f1-score'] * 100
stop_f1 = report['stop']['f1-score'] * 100

# Print the results in percentage format
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision for 'on': {on_precision:.2f}%")
print(f"Precision for 'stop': {stop_precision:.2f}%")
print(f"Recall for 'on': {on_recall:.2f}%")
print(f"Recall for 'stop': {stop_recall:.2f}%")
print(f"F1-Score for 'on': {on_f1:.2f}%")
print(f"F1-Score for 'stop': {stop_f1:.2f}%")
import joblib

# Save the trained model to a file
joblib.dump(classifier, 'speech_command_model.pkl')
print("Model saved to 'speech_command_model.pkl'")


