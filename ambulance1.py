import os
import time
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import threading
import queue
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model

# Constants
SAMPLE_RATE = 22050  # Sample rate in Hz (same as used in training)
FRAME_SIZE = 2048  # Audio frame size
HOP_LENGTH = 512  # Hop length for overlapping frames
DURATION = 2  # Duration of audio to analyze in seconds
OVERLAP = 1  # Overlap between consecutive audio segments in seconds
DETECTION_THRESHOLD = 0.75  # Confidence threshold for ambulance siren detection

# Load the pre-trained model
MODEL_PATH = 'ambulance_siren_detector.h5'  # Path to your saved model

# Function to extract features from audio data (same as in training script)
def extract_features(audio, sample_rate=SAMPLE_RATE, n_mfcc=13, n_mels=128, frames=128):
    try:
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
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

# Class for real-time audio processing
class RealtimeSirenDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=DETECTION_THRESHOLD):
        self.model = load_model(model_path)
        self.threshold = threshold
        self.audio_queue = queue.Queue()
        self.detection_history = []
        self.spectrogram_data = None
        self.is_running = False
        self.plot_initialized = False
        
        # Load alert sound
        self.alert_sound = AudioSegment.from_file("alert.wav") if os.path.exists("alert.wav") else None
        if self.alert_sound is None:
            print("Alert sound file not found. No alert will be played.")
        
        # Create figure for visualization
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Real-time Ambulance Siren Detector')
        
        # Initialize history plot
        self.detection_line, = self.ax[0].plot([], [], 'r-')
        self.ax[0].set_ylim(0, 1)
        self.ax[0].set_xlim(0, 50)  # Show last 50 frames
        self.ax[0].set_title('Ambulance Siren Detection Probability')
        self.ax[0].set_ylabel('Probability')
        self.ax[0].set_xlabel('Time (frames)')
        self.ax[0].axhline(y=self.threshold, color='g', linestyle='--', label=f'Threshold ({self.threshold})')
        self.ax[0].legend()
        
        # Initialize spectrogram plot
        self.spec_plot = self.ax[1].imshow(np.zeros((128, 128)), aspect='auto', origin='lower', 
                                           vmin=-80, vmax=0, cmap='inferno')
        self.ax[1].set_title('Mel Spectrogram')
        self.ax[1].set_ylabel('Mel Frequency')
        self.ax[1].set_xlabel('Time Frames')
        plt.colorbar(self.spec_plot, ax=self.ax[1], format='%+2.0f dB')
        
        plt.tight_layout()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio capture"""
        if status:
            print(f"Status: {status}")
        
        # Convert from float32 to int16
        audio_data = indata[:, 0]  # Get the first channel
        
        # Add to queue
        self.audio_queue.put(audio_data)
    
    def process_audio(self):
        """Process audio chunks and make predictions"""
        buffer = np.array([])
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get(timeout=1)
                    
                    # Add to buffer
                    buffer = np.append(buffer, chunk)
                    
                    # Keep only the latest DURATION seconds of audio
                    buffer_size = int(SAMPLE_RATE * DURATION)
                    if len(buffer) > buffer_size:
                        buffer = buffer[-buffer_size:]
                    
                    # Only process if we have enough data
                    if len(buffer) >= buffer_size:
                        # Extract features
                        features = extract_features(buffer)
                        
                        if features is not None:
                            # Save for visualization
                            self.spectrogram_data = features
                            
                            # Reshape for model input
                            features = features.reshape(1, features.shape[0], features.shape[1], 1)
                            
                            # Make prediction
                            prediction = self.model.predict(features, verbose=0)[0]
                            ambulance_prob = prediction[1]  # Probability of ambulance class
                            
                            # Add to history
                            self.detection_history.append(ambulance_prob)
                            if len(self.detection_history) > 100:  # Keep last 100 predictions
                                self.detection_history = self.detection_history[-100:]
                            
                            # Check if probability exceeds threshold
                            if ambulance_prob >= self.threshold:
                                print(f"AMBULANCE DETECTED! Confidence: {ambulance_prob:.2f}")
                                
                                # Play alert sound if available and not already playing
                                if self.alert_sound is not None:
                                    threading.Thread(target=play, args=(self.alert_sound,)).start()
                            else:
                                print(f"Regular traffic. Ambulance probability: {ambulance_prob:.2f}")
                            
                            # Force buffer to overlap with current data (sliding window)
                            overlap_samples = int(OVERLAP * SAMPLE_RATE)
                            buffer = buffer[-overlap_samples:]
                
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
    
    def update_plot(self, frame):
        """Update function for matplotlib animation"""
        # Update detection history plot
        x = np.arange(len(self.detection_history))
        self.detection_line.set_data(x, self.detection_history)
        
        if len(self.detection_history) > 0:
            self.ax[0].set_xlim(max(0, len(self.detection_history) - 50), max(50, len(self.detection_history)))
        
        # Update spectrogram plot
        if self.spectrogram_data is not None:
            self.spec_plot.set_array(self.spectrogram_data)
        
        return self.detection_line, self.spec_plot
    
    def start(self):
        """Start the detector"""
        if self.is_running:
            print("Detector is already running")
            return
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            interval=100, blit=True
        )
        plt.show()
    
    def stop(self):
        """Stop the detector"""
        self.is_running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        
        plt.close(self.fig)
        print("Detector stopped")

if __name__ == "__main__":
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found.")
            print("Please make sure you've trained the model first.")
            exit(1)
        
        print("Starting real-time ambulance siren detector...")
        print(f"Detection threshold: {DETECTION_THRESHOLD}")
        print("Press Ctrl+C to stop")
        
        # Create and start detector
        detector = RealtimeSirenDetector()
        detector.start()
        
    except KeyboardInterrupt:
        print("\nStopping detector...")
        if 'detector' in locals():
            detector.stop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'detector' in locals():
            detector.stop()