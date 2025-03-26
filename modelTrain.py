import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ✅ Define Dataset Paths
DATASET_PATH = "dataset/"
SIREN_PATH = os.path.join(DATASET_PATH, "ambulance")  
TRAFFIC_PATH = os.path.join(DATASET_PATH, "traffic")  

# ✅ Function to Extract Mel Spectrograms with Augmentation
def extract_features(file_path, max_pad_len=128, add_noise=False):
    y, sr = librosa.load(file_path, sr=22050)

    # Apply noise augmentation
    if add_noise:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise  

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_len]

    return mel_spec_db

# ✅ Load Dataset with Augmentation
X, y = [], []
for file in os.listdir(SIREN_PATH):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(SIREN_PATH, file), add_noise=True)  
        X.append(features)
        y.append(1)  

for file in os.listdir(TRAFFIC_PATH):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(TRAFFIC_PATH, file), add_noise=True)  
        X.append(features)
        y.append(0)  

X = np.array(X)[..., np.newaxis]  # Add channel dimension
y = np.array(y)

# ✅ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build Improved CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 128, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ✅ Train Model with More Epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# ✅ Save the Improved Model
model.save("ambulance_siren_model_v2.h5")

print("✅ Training complete! Model saved as 'ambulance_siren_model_v2.h5'.")
