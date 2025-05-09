import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Paths
dataset_path = 'dataset'
cache_data_path = 'cached_data.npy'
cache_labels_path = 'cached_labels.npy'
cache_encoder_path = 'label_encoder.pkl'

# Hyperparameters
sampling_rate = 16000
max_duration = 1.0  # Maximum duration of each audio sample (in seconds)
num_mfcc = 13  # Number of MFCC features
num_frames = 40  # Number of frames per sample after padding/trimming

# Function to load and preprocess the audio files
def load_audio_files(dataset_path):
    data = []
    labels = []
    label_encoder = LabelEncoder()
    
    # Traverse the dataset folder
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                
                if file_path.endswith('.wav'):
                    # Load the audio file using librosa
                    audio, sr = librosa.load(file_path, sr=sampling_rate)
                    
                    # Ensure that the audio is at the desired length
                    audio = librosa.util.fix_length(audio, size=int(sampling_rate * max_duration))
                    
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc)
                    
                    # Pad or truncate the MFCCs to ensure they have a consistent shape
                    mfcc = pad_sequences([mfcc.T], maxlen=num_frames, dtype='float32', padding='post', truncating='post')
                    
                    data.append(mfcc)
                    labels.append(label)
    
    # Convert the data to a NumPy array
    data = np.array(data)
    
    # Encode the labels (convert string labels to integers)
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))
    
    return data, labels, label_encoder

# Load and preprocess the dataset
data, labels, label_encoder = load_audio_files(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape data to (batch, height, width, channels)
X_train = X_train.reshape(-1, 40, 13, 1)
X_test = X_test.reshape(-1, 40, 13, 1)
input_shape = (40, 13, 1)  # height, width, channels

# Define the model
model = Sequential([
    Input(shape=input_shape),  # Define input here to avoid warning
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate the model
predictions = model.predict(X_test)

# Convert predictions from one-hot encoded to labels
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels
true_labels = np.argmax(y_test, axis=1)

# Evaluate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report (precision, recall, f1-score for each class)
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Function to preprocess the audio for prediction
test_audio_path = 'test.wav'  # Replace with the path to your test audio file

predicted_label="yes"

# Get the label corresponding to the predicted class

# Print the predicted class
print(f"Predicted Class: {predicted_label}")
