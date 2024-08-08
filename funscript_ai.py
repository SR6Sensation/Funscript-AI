import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

# Load .funscript file
def load_funscript(funscript_path):
    with open(funscript_path, 'r') as file:
        funscript = json.load(file)
    return funscript

# Extract frames from .mp4 file
def extract_frames(mp4_path, fps=30):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    timestamps = []
    success, frame = cap.read()
    count = 0
    while success:
        if count % fps == 0:  # Adjust to get frame at specific intervals
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert to grayscale
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        success, frame = cap.read()
        count += 1
    cap.release()
    return np.array(frames), np.array(timestamps)

# Synchronize frames with funscript actions
def synchronize_data(frames, timestamps, funscript):
    actions = funscript['actions']
    action_times = [action['at'] for action in actions]
    action_positions = [action['pos'] for action in actions]

    # Interpolate positions for each frame timestamp
    positions = np.interp(timestamps, action_times, action_positions)
    
    return frames, positions

# Prepare data for LSTM
def prepare_data(frames, positions, seq_length=30):
    X, y = [], []
    for i in range(len(frames) - seq_length):
        X.append(frames[i:i+seq_length])
        y.append(positions[i:i+seq_length])
    return np.array(X), np.array(y)

# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model

# Generate .funscript from model predictions
def generate_funscript(predictions, timestamps):
    actions = [{"at": int(ts), "pos": int(pos)} for ts, pos in zip(timestamps, predictions)]
    funscript = {
        "actions": actions,
        "inverted": False,
        "metadata": {
            "bookmarks": [],
            "chapters": [],
            "creator": "",
            "description": "",
            "duration": int(timestamps[-1]),
            "license": "",
            "notes": "",
            "performers": [],
            "script_url": "",
            "tags": [],
            "title": "",
            "type": "basic",
            "video_url": ""
        },
        "range": 100,
        "version": "1.0"
    }
    return funscript

# Save funscript to file
def save_funscript(funscript, output_path):
    with open(output_path, 'w') as file:
        json.dump(funscript, file)

# Save model to file
def save_model(model, model_path):
    model.save(model_path)

# Load model from file
def load_existing_model(model_path):
    return load_model(model_path)

# Main script
def main(mp4_path, funscript_path, output_funscript_path, model_path=None, new_model_path='model.h5', epochs=10):
    funscript = load_funscript(funscript_path)
    frames, timestamps = extract_frames(mp4_path)
    frames, positions = synchronize_data(frames, timestamps, funscript)
    X, y = prepare_data(frames, positions)
    
    if model_path and os.path.exists(model_path):
        model = load_existing_model(model_path)
    else:
        model = build_model((X.shape[1], X.shape[2]))
    
    model = train_model(model, X, y, epochs=epochs)
    save_model(model, new_model_path)
    
    predictions = model.predict(X)
    predictions = predictions.flatten()
    
    new_funscript = generate_funscript(predictions, timestamps[:len(predictions)])
    save_funscript(new_funscript, output_funscript_path)

if __name__ == "__main__":
    # Initial training:     main('path/to/video.mp4', 'path/to/original.funscript', 'path/to/generated.funscript', new_model_path='path/to/save/new/model.h5', epochs=10)
    # Incremental training: main('path/to/new/video.mp4', 'path/to/new.funscript', 'path/to/new/generated.funscript', model_path='path/to/saved/model.h5', new_model_path='path/to/save/updated/model.h5', epochs=10)

    main('path/to/video.mp4', 'path/to/original.funscript', 'path/to/generated.funscript', model_path='path/to/saved/model.h5', new_model_path='path/to/save/new/model.h5', epochs=10)
