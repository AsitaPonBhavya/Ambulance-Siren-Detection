import numpy as np
import sounddevice as sd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class SirenDirectionDetector:
    def __init__(self, sample_rate=44100, chunk_size=1024, left_channel_gain=1.0, right_channel_gain=5.0):
        """
        Initialize the siren direction detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            left_channel_gain: Gain to apply to left channel (MacBook mic)
            right_channel_gain: Gain to apply to right channel (other mic)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.left_gain = left_channel_gain
        self.right_gain = right_channel_gain
        
        # Audio buffer to store incoming audio
        self.audio_buffer = np.zeros((chunk_size, 2), dtype=np.float32)
        
        # Siren frequency range (typical for emergency vehicles)
        self.siren_freq_low = 600
        self.siren_freq_high = 1200
        
        # For visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.direction_indicator = self.ax.text(0.5, 0.5, '', 
                                               horizontalalignment='center',
                                               verticalalignment='center',
                                               fontsize=24)
        self.left_power_text = self.ax.text(0.1, 0.9, 'Left: 0', fontsize=14)
        self.right_power_text = self.ax.text(0.7, 0.9, 'Right: 0', fontsize=14)
        self.power_diff_text = self.ax.text(0.4, 0.8, 'Diff: 0', fontsize=14)
        
        # Create a simple visualization
        self.arrow, = self.ax.plot([], [], 'r-', linewidth=3)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title('Siren Direction')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Print available audio devices
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            channels_info = f"({dev['max_input_channels']} in, {dev['max_output_channels']} out)"
            print(f"  {i} {dev['name']}, {channels_info}")
        
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for the audio stream.
        """
        if status:
            print(f"Status: {status}")
        self.audio_buffer = indata.copy()
        
    def detect_siren_in_channel(self, audio_data):
        """
        Detect if there's a siren sound in the audio data.
        Returns the power of the siren frequency range.
        """
        # Apply bandpass filter for siren frequency range
        sos = signal.butter(10, [self.siren_freq_low, self.siren_freq_high], 
                           'bandpass', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, audio_data)
        
        # Calculate power in the siren frequency range
        power = np.mean(filtered**2)
        
        return power
    
    def update_direction_indicator(self, i):
        """
        Update the direction indicator based on audio input.
        """
        # Get audio data from buffer
        audio_data = self.audio_buffer
        
        # Split stereo channels
        left_channel = audio_data[:, 0]  # First mic channel
        right_channel = audio_data[:, 1]  # Second mic channel
        
        # Apply gain correction
        left_channel = left_channel / self.left_gain
        right_channel = right_channel / self.right_gain
        
        # Detect siren in each channel
        left_power = self.detect_siren_in_channel(left_channel)
        right_power = self.detect_siren_in_channel(right_channel)
        
        # Determine direction based on power difference
        power_diff = left_power - right_power
        
        # Create arrow direction
        if abs(power_diff) < 0.0001:  # Threshold for "straight ahead"
            arrow_x = [0, 0]
            arrow_y = [0, 1]
            direction_text = "AHEAD"
        else:
            angle = np.arctan2(power_diff, 1.0)  # Convert power difference to angle
            arrow_x = [0, np.cos(angle)]
            arrow_y = [0, np.sin(angle)]
            
            if power_diff > 0:
                direction_text = "LEFT"
            else:
                direction_text = "RIGHT"
        
        # Update visualization
        self.arrow.set_data(arrow_x, arrow_y)
        self.direction_indicator.set_text(direction_text)
        self.left_power_text.set_text(f'Left: {left_power:.6f}')
        self.right_power_text.set_text(f'Right: {right_power:.6f}')
        self.power_diff_text.set_text(f'Diff: {power_diff:.6f}')
        
        return self.arrow, self.direction_indicator, self.left_power_text, self.right_power_text, self.power_diff_text
    
    def find_dual_mic_device(self):
        """
        Find a device with at least 2 input channels.
        """
        devices = sd.query_devices()
        dual_mic_devices = []
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] >= 2:
                dual_mic_devices.append((i, dev))
                
        if not dual_mic_devices:
            print("No devices with dual microphone input found.")
            return None
            
        print("\nDevices with at least 2 input channels:")
        for i, (idx, dev) in enumerate(dual_mic_devices):
            print(f"{i}: {dev['name']} (Input channels: {dev['max_input_channels']})")
        
        if len(dual_mic_devices) == 1:
            print(f"Auto-selecting the only dual-mic device: {dual_mic_devices[0][1]['name']}")
            return dual_mic_devices[0][0]
        
        selection = int(input("Enter the number of the dual-mic device to use: "))
        return dual_mic_devices[selection][0]
    
    def run(self):
        """
        Run the detector with visualization.
        """
        # Find dual mic device
        device_id = self.find_dual_mic_device()
        if device_id is None:
            print("No suitable device found. Exiting.")
            return
        
        # Start audio stream
        with sd.InputStream(device=device_id, 
                            channels=2, 
                            callback=self.audio_callback,
                            samplerate=self.sample_rate,
                            blocksize=self.chunk_size):
            
            print(f"Using device: {sd.query_devices(device=device_id)['name']}")
            print("Starting siren direction detector. Close the plot window to exit.")
            
            # Start animation
            ani = FuncAnimation(self.fig, self.update_direction_indicator, interval=100, blit=True)
            plt.show()

if __name__ == "__main__":
    # Create detector
    detector = SirenDirectionDetector(left_channel_gain=1.0, right_channel_gain=1.0)
    
    try:
        detector.run()
    except Exception as e:
        print(f"Error: {e}")