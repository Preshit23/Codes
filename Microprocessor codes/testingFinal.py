import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from scipy.signal import butter, filtfilt
import numpy as np
from adafruit_servokit import ServoKit
import pickle

# Loading the trained SVM model
with open('svm_rbf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

###### CONFIGURATION ######

i2c_bus = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus)
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
ads.reference_voltage = 1  
ads.gain = 8
ads.data_rate = 860
emg_chan = AnalogIn(ads, ADS.P0)
pressure_chan = AnalogIn(ads, ADS.P1)

low_cut = 50  # Lower cutoff frequency (slightly lower than the original frequency)
high_cut = 400  # Upper cutoff frequency (slightly higher than the original frequency)
fs = 1000  # Sampling frequency
order = 5  # filter order

###### FUNCTIONS ######

# Function to control servo motors
def control_motors(angle):
    for i in range(5):
        kit.servo[i].angle = angle

# Apply bandpass filter to a signal
def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Rectify the signal
def rectify_signal(signal):
    return np.abs(signal)

# Extract features from a chunk
def extract_features(chunk_signal):
    average_amplitude = np.mean(chunk_signal)
    rms = np.sqrt(np.mean(chunk_signal**2))
    slope = np.mean(np.diff(chunk_signal))
    if np.isnan(slope):
        slope = 0
    return [average_amplitude, rms, slope]


####### MAIN PROGRAM ######


try:
    while True:
        # Collect a chunk of 5 data points
        raw_data = []
        for _ in range(5):
            raw_value = emg_chan.value
            #voltage = raw_value * 3.3 / 65535
            raw_data.append(raw_value)
            time.sleep(0.1)

        # Apply bandpass filter to the collected data
        filtered_data = bandpass_filter(raw_data, low_cut, high_cut, fs, order)

        # Rectify the filtered signal
        rectified_data = rectify_signal(filtered_data)

        # Extract features from the rectified signal
        test_features = extract_features(rectified_data)

        # Reshape features for model prediction
        test_features = np.array(test_features).reshape(1, -1)

        # Use the loaded SVM model to predict
        prediction = loaded_model.predict(test_features)

        # Perform actions based on the prediction
        if prediction == 0:  # Open hand movement
            control_motors(0)  # Set servo angles for open hand movement
        elif prediction == 1:  # Close hand movement
            control_motors(180)  # Set servo angles for close hand movement
          
        pressure = pressure_chan.value
        if pressure > 2000:
          print("TOUCH FEEDBACK DETECTED")
          
except KeyboardInterrupt:
    pass
