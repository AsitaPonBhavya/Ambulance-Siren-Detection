# Ambulance Siren Detection and Traffic Signal Automation

This project implements a **real-time ambulance siren detection system** using a **CNN model** and automates traffic signals based on the detected siren. The system can **differentiate between ambulance sirens and normal traffic sounds** such as car horns or background noise.

## **Features**
- Detects if an incoming sound is an ambulance siren or normal traffic noise.
- Uses a **trained Convolutional Neural Network (CNN)** for classification.
- Can **automatically change traffic signals** when an ambulance is detected.
- Filters out other noises like honks and general traffic sounds.
- Uses **multiple microphones** to detect the **direction of the ambulance** and turns the **traffic signal green** from that direction.

---

## **Hardware Components**
| Component        | Purpose |
|-----------------|---------|
| MAX9814 Electret Microphone | Captures incoming audio signals |
| ESP8266 Microcontroller | Processes and sends signals to control traffic lights |
| Relay Module | Switches traffic lights based on ambulance detection |
| Breadboard & Jumper Wires | Circuit setup for connections |

---

## **Software & Algorithms**
- **Convolutional Neural Network (CNN)**: Used to classify sound as either "Ambulance Siren" or "Traffic Noise."
- **Fast Fourier Transform (FFT)**: Extracts frequency features from audio input.
- **Python & Arduino IDE**: Used for model training and ESP8266 microcontroller programming.

---

## **Dataset & Model Training**
We trained our CNN model using two datasets:
1. **Ambulance Siren Dataset** (`ambulance/` folder) - Contains multiple ambulance siren audio files.
2. **Traffic Noise Dataset** (`traffic/` folder) - Contains normal traffic sounds like honking, engine noise, and background chatter.

### **Model Architecture**
The model consists of:
- 3 **Convolutional Layers** with MaxPooling
- Fully connected **Dense Layers**
- Dropout for regularization
- Output Layer with **2 Classes**: `Ambulance` and `Traffic`

#### **Training Performance**
- **Accuracy:** ~95% on test data
- **Loss:** Reduced significantly after initial epochs
- **Confusion Matrix Results:** High true positive rate for both classes

---

## **Implementation Steps**
1. **Connect MAX9814 microphones** to ESP8266 for real-time sound capture.
2. Preprocess the captured audio using **FFT** and send it to the CNN model.
3. If an **ambulance siren** is detected, **ESP8266 activates the relay module**.
4. Relay module **changes the traffic light** from the **ambulance's direction to green**.

---

## **Future Enhancements**
- **Multi-Microphone Setup**: Use **multiple microphones** to estimate ambulance direction and dynamically adjust signals.
- **Edge AI Deployment**: Run CNN on ESP32 with **TensorFlow Lite** for real-time processing.
- **Cloud Integration**: Send real-time ambulance detections to cloud servers for traffic control.

---

## **Project Contributors**
- **[Your Name]** - AI Model Development & Training
- **[Other Team Members]** - Embedded System Design & Signal Processing

---

## **How to Run the Project**
1. **Train the model** using `train_model.py`
2. **Deploy to ESP8266** using `arduino_code.ino`
3. **Test with real-world audio samples** or use `test_audio.py`


---

🚀 **This project aims to reduce ambulance delays and improve emergency response times using AI and IoT!** 🚀
