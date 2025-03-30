import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to extract features from audio files
def extract_features(file_path, n_mfcc=13, n_mels=128, frames=128):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate MFCCs to ensure consistent dimensions
        if mfccs.shape[1] > frames:
            mfccs = mfccs[:, :frames]
        else:
            pad_width = frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize mel spectrogram
        if mel_spec_db.shape[1] > frames:
            mel_spec_db = mel_spec_db[:, :frames]
        else:
            pad_width = frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        return mfccs, mel_spec_db
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

# Data preparation function
def prepare_data(data_dir, use_mel=True):
    features = []
    labels = []
    file_paths = []
    
    # Loop through each class directory
    for label, class_name in enumerate(['traffic', 'ambulance']):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                mfccs, mel_spec = extract_features(file_path)
                
                if mfccs is not None and mel_spec is not None:
                    # Use either MFCCs or mel spectrogram based on parameter
                    if use_mel:
                        features.append(mel_spec)
                    else:
                        features.append(mfccs)
                    
                    labels.append(label)
                    file_paths.append(file_path)
    
    # Convert lists to arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Add channel dimension for CNN input
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels)
    
    return features, labels, file_paths

# Function to build CNN model
def build_model(input_shape):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten the output and feed it into dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 classes: traffic and ambulance
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train the model
def train_siren_detector(data_dir, use_mel=True, test_size=0.2, epochs=20, batch_size=32):
    # Prepare data
    print("Extracting features from audio files...")
    features, labels, file_paths = prepare_data(data_dir, use_mel)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Build model
    model = build_model(input_shape=X_train[0].shape)
    print(model.summary())
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    class_names = ['Traffic', 'Ambulance']
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Label the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save the model
    model.save('ambulance_siren_detector.h5')
    print("Model saved as 'ambulance_siren_detector.h5'")
    
    return model, history

# Function to make predictions on new audio
def predict_audio(model, file_path, use_mel=True):
    mfccs, mel_spec = extract_features(file_path)
    
    if mfccs is None or mel_spec is None:
        return None
    
    # Select feature type based on how the model was trained
    if use_mel:
        features = mel_spec
    else:
        features = mfccs
    
    # Reshape for model input
    features = features.reshape(1, features.shape[0], features.shape[1], 1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    class_names = ['Traffic', 'Ambulance']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]
    
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(prediction)}
    }

# Example usage
if __name__ == "__main__":
    # Set the path to your dataset
    data_dir = "dataset"  
    
    # Train the model
    model, history = train_siren_detector(
        data_dir=data_dir,
        use_mel=True,  # Use mel spectrograms instead of MFCCs
        epochs=20,
        batch_size=32
    )
    
    # Example: Make prediction on a new audio file
    test_file = "dataset/ambulance/mixed_sound_8.wav" 
    if os.path.exists(test_file):
        result = predict_audio(model, test_file, use_mel=True)
        print(f"Prediction: {result['predicted_class']} with {result['confidence']*100:.2f}% confidence")
