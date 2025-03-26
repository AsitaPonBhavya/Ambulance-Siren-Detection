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
SAMPLE_RATE = 22050  # Sample rate in Hz
FRAME_SIZE = 2048    # Audio frame size
DURATION = 2         # Duration of audio to analyze in seconds
OVERLAP = 1          # Overlap between consecutive audio segments
DETECTION_THRESHOLD = 0.99  # Confidence threshold for siren detection
DIRECTION_THRESHOLD = 0.05  # Threshold for determining direction
MODEL_PATH = 'ambulance_siren_detector.h5'

# Function to extract features from audio data
def extract_features(audio, sample_rate=SAMPLE_RATE, n_mels=128, frames=128):
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max(0, frames - mel_spec_db.shape[1]))), mode='constant')
        return mel_spec_db
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

class SimpleSirenDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=DETECTION_THRESHOLD):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
        
        self.threshold = threshold
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.current_status = "NO SIREN"
        self.current_direction = "N/A"
        self.siren_confidence = 0.0
        self.direction_value = 0.0
        self.alert_sound = AudioSegment.from_file("alert.wav") if os.path.exists("alert.wav") else None
        self.last_alert_time = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.axis('off')
        self.status_text = self.fig.text(0.5, 0.6, "Status: NO SIREN DETECTED", ha='center', fontsize=24, fontweight='bold')
        self.direction_text = self.fig.text(0.5, 0.4, "Direction: N/A", ha='center', fontsize=24, fontweight='bold')
        self.confidence_text = self.fig.text(0.5, 0.25, "Confidence: 0%", ha='center', fontsize=18)
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        try:
            if indata.shape[1] >= 2:
                left_channel, right_channel = indata[:, 0], indata[:, 1]
                left_energy, right_energy = np.sum(np.abs(left_channel)), np.sum(np.abs(right_channel))
                mono_mix = (left_channel + right_channel) / 2
                self.audio_queue.put((mono_mix, left_energy, right_energy))
            else:
                self.audio_queue.put((indata[:, 0], 0, 0))
                print("Warning: Mono input detected. Direction detection may not work.")
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def process_audio(self):
        buffer = np.array([])
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    chunk, left_energy, right_energy = self.audio_queue.get()
                    buffer = np.append(buffer, chunk)[-int(SAMPLE_RATE * DURATION):]
                    if len(buffer) >= SAMPLE_RATE * DURATION:
                        features = extract_features(buffer)
                        if features is not None:
                            features = features.reshape(1, features.shape[0], features.shape[1], 1)
                            ambulance_prob = float(self.model.predict(features, verbose=0)[0][1])
                            self.siren_confidence = ambulance_prob
                            direction = (right_energy - left_energy) / max(1, (left_energy + right_energy))
                            self.direction_value = 0.3 * direction + 0.7 * getattr(self, 'prev_direction', 0)
                            self.prev_direction = self.direction_value
                            
                            if ambulance_prob >= self.threshold:
                                self.current_status = "SIREN DETECTED"
                                self.current_direction = "LEFT" if self.direction_value < -DIRECTION_THRESHOLD else "RIGHT" if self.direction_value > DIRECTION_THRESHOLD else "FRONT"
                                if self.alert_sound and time.time() - self.last_alert_time > 2:
                                    threading.Thread(target=play, args=(self.alert_sound,)).start()
                                    self.last_alert_time = time.time()
                            else:
                                self.current_status, self.current_direction = "NO SIREN", "N/A"
                time.sleep(0.1)
            except Exception as e:
                print(f"Processing error: {e}")
    
    def update_display(self, frame):
        self.status_text.set_text(f"Status: {self.current_status}")
        self.direction_text.set_text(f"Direction: {self.current_direction}")
        self.confidence_text.set_text(f"Confidence: {self.siren_confidence * 100:.1f}%")
        return self.status_text, self.direction_text, self.confidence_text
    
    def start(self):
        if self.is_running:
            print("Detector is already running")
            return
        self.is_running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, channels=2, callback=self.audio_callback)
        self.stream.start()
        self.ani = FuncAnimation(self.fig, self.update_display, interval=100, blit=False)
        plt.show()
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        plt.close(self.fig)
        print("Detector stopped")

if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found.")
            exit(1)
        detector = SimpleSirenDetector()
        detector.start()
    except KeyboardInterrupt:
        detector.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        detector.stop()
