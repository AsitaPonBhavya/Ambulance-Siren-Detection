import os
import time
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import threading
import queue
import serial
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
DIRECTION_THRESHOLD = 0.05  # Threshold for determining direction (difference between L/R channels)

# Load the pre-trained model
MODEL_PATH = 'ambulance_siren_detector.h5'  # Path to your saved model

# Function to extract features from audio data (same as in training script)
def extract_features(audio, sample_rate=SAMPLE_RATE, n_mels=128, frames=128):
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

class DirectionalSirenDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=DETECTION_THRESHOLD):
        self.model = load_model(model_path)
        self.threshold = threshold
        self.audio_queue = queue.Queue()
        self.detection_history = []
        self.direction_history = []  # -1 for left, 0 for center, 1 for right
        self.spectrogram_data = None
        self.is_running = False
        self.last_sent_signal = None  # Track last ESP8266 signal
        # Load alert sound
        self.esp = None
        self.alert_sound = AudioSegment.from_file("alert.wav") if os.path.exists("alert.wav") else None
        try:
            self.esp = serial.Serial("/dev/tty.SLAB_USBtoUART", 115200, timeout=1)
            time.sleep(2)  # Allow connection to establish
            print("Connected to ESP8266")
        except serial.SerialException as e:
            print(f"Warning: Serial connection failed ({e}). ESP8266 control disabled.")
        if self.alert_sound is None:
            print("Alert sound file not found. No alert will be played.")
        
        # Create figure for visualization
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.canvas.manager.set_window_title('Directional Ambulance Siren Detector')
        
        # Initialize detection probability plot
        self.detection_line, = self.ax[0].plot([], [], 'r-')
        self.ax[0].set_ylim(0, 1)
        self.ax[0].set_xlim(0, 50)  # Show last 50 frames
        self.ax[0].set_title('Ambulance Siren Detection Probability')
        self.ax[0].set_ylabel('Probability')
        self.ax[0].axhline(y=self.threshold, color='g', linestyle='--', label=f'Threshold ({self.threshold})')
        self.ax[0].legend()
        
        # Initialize direction plot
        self.direction_line, = self.ax[1].plot([], [], 'b-')
        self.ax[1].set_ylim(-1.1, 1.1)
        self.ax[1].set_xlim(0, 50)  # Show last 50 frames
        self.ax[1].set_title('Siren Direction (-1: Left, 0: Center, 1: Right)')
        self.ax[1].set_ylabel('Direction')
        self.ax[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        self.ax[1].axhline(y=DIRECTION_THRESHOLD, color='g', linestyle='--', alpha=0.5)
        self.ax[1].axhline(y=-DIRECTION_THRESHOLD, color='g', linestyle='--', alpha=0.5)
        
        # Initialize spectrogram plot
        self.spec_plot = self.ax[2].imshow(np.zeros((128, 128)), aspect='auto', origin='lower', 
                                           vmin=-80, vmax=0, cmap='inferno')
        self.ax[2].set_title('Mel Spectrogram')
        self.ax[2].set_ylabel('Mel Frequency')
        self.ax[2].set_xlabel('Time Frames')
        plt.colorbar(self.spec_plot, ax=self.ax[2], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Text annotations for current status
        self.status_text = self.fig.text(0.5, 0.01, "Status: Monitoring...", ha='center', fontsize=12)
        self.direction_text = self.fig.text(0.5, 0.04, "Direction: N/A", ha='center', fontsize=12, 
                                           bbox=dict(facecolor='white', alpha=0.5))
    def send_esp_signal(self, message):
        """Send a command to ESP8266 only if it's different from the last signal."""
        if self.esp and self.esp.is_open and message != self.last_sent_signal:
            self.esp.write(f"{message}\n".encode())
            self.last_sent_signal = message
            print(f"Sent to ESP8266: {message}")   
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio capture"""
        if status:
            print(f"Status: {status}")
        
        # Get stereo channels
        if indata.shape[1] >= 2:
            left_channel = indata[:, 0]
            right_channel = indata[:, 1]
            
            # Calculate channel energies for direction detection
            left_energy = np.sum(np.abs(left_channel)) / len(left_channel)
            right_energy = np.sum(np.abs(right_channel)) / len(right_channel)
            
            # Mono mix for siren detection (as trained model expects mono)
            mono_mix = (left_channel + right_channel) / 2
            
            # Add to queue with channel energy information
            self.audio_queue.put((mono_mix, left_energy, right_energy))
        else:
            # Handle mono input (no direction detection)
            mono_channel = indata[:, 0]
            self.audio_queue.put((mono_channel, 0, 0))
            print("Warning: Mono input detected. Direction detection requires stereo input.")
    
    
    def process_audio(self):
        """Process audio chunks and make predictions"""
        buffer = np.array([])

        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    chunk, left_energy, right_energy = self.audio_queue.get(timeout=1)
                    buffer = np.append(buffer, chunk)
                    buffer_size = int(SAMPLE_RATE * DURATION)

                    if len(buffer) > buffer_size:
                        buffer = buffer[-buffer_size:]

                    if len(buffer) >= buffer_size:
                        features = extract_features(buffer)
                        if features is not None:
                            features = features.reshape(1, features.shape[0], features.shape[1], 1)
                            prediction = self.model.predict(features, verbose=0)[0]
                            ambulance_prob = prediction[1]  

                            self.detection_history.append(ambulance_prob)
                            self.detection_history = self.detection_history[-100:]

                            if left_energy > 0 or right_energy > 0:
                                energy_sum = left_energy + right_energy
                                direction = (right_energy - left_energy) / energy_sum
                                direction = 0.3 * direction + 0.7 * (self.direction_history[-1] if self.direction_history else 0)
                                self.direction_history.append(direction)
                                self.direction_history = self.direction_history[-100:]

                            # Handle detection logic
                            if ambulance_prob >= self.threshold:
                                if abs(direction) < DIRECTION_THRESHOLD:
                                    direction_text = "FRONT"
                                elif direction < 0:
                                    direction_text = "LEFT"
                                else:
                                    direction_text = "RIGHT"

                                print(f"AMBULANCE DETECTED! Direction: {direction_text} (Confidence: {ambulance_prob:.2f})")

                                if direction_text == "RIGHT":
                                    self.send_esp_signal("AMBULANCE")  # Green ON only for LEFT
                                else:
                                    self.send_esp_signal("NORMAL")  # Keep Red ON for FRONT and RIGHT
                            else:
                                self.send_esp_signal("NORMAL")  # No ambulance detected

                            buffer = buffer[-int(OVERLAP * SAMPLE_RATE):]

                time.sleep(0.1)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")

                
    def update_plot(self, frame):
        """Update function for matplotlib animation"""
        # Update detection history plot
        x = np.arange(len(self.detection_history))
        self.detection_line.set_data(x, self.detection_history)
        
        # Update direction history plot
        x_dir = np.arange(len(self.direction_history))
        self.direction_line.set_data(x_dir, self.direction_history)
        
        if len(self.detection_history) > 0:
            xlim_max = max(50, len(self.detection_history))
            xlim_min = max(0, xlim_max - 50)
            self.ax[0].set_xlim(xlim_min, xlim_max)
            self.ax[1].set_xlim(xlim_min, xlim_max)
        
        # Update spectrogram plot
        if self.spectrogram_data is not None:
            self.spec_plot.set_array(self.spectrogram_data)
        
        return self.detection_line, self.direction_line, self.spec_plot

    
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
        
        # Start audio stream (stereo input)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=2,  # Stereo input
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
        """Stop the detector and clean up resources"""
        self.is_running = False
        if self.esp and self.esp.is_open:
            self.esp.close()
            print("Closed serial connection to ESP8266")

if __name__ == "__main__":
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found.")
            print("Please make sure you've trained the model first.")
            exit(1)
        
        print("Starting directional ambulance siren detector...")
        print(f"Detection threshold: {DETECTION_THRESHOLD}")
        print(f"Direction threshold: {DIRECTION_THRESHOLD}")
        print("Listening for stereo input to determine siren direction...")
        print("Press Ctrl+C to stop")
        
        # Create and start detector
        detector = DirectionalSirenDetector()
        detector.start()
        
    except KeyboardInterrupt:
        print("\nStopping detector...")
        if 'detector' in locals():
            detector.stop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'detector' in locals():
            detector.stop()