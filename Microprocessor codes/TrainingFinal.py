# MODULE LIBRARIES
import time
import board
import adafruit_ads1x15.ads1115 as ADS
import busio
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_blinka.board.raspberrypi.raspi_40pin as pin
# FILTER LIBRARIES
import numpy as np
from scipy.signal import butter, filtfilt
# PLOT LIBRARIES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# SVM LIBRARIES
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# PICKLE
import pickle

###### CONFIGURATION ######
# I2C bus for ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
# ADC object using the I2C bus
ads = ADS.ADS1115(i2c)
# ADS Config
GAIN = 8
referenceV = 1
SAMPLING_RATE = 860  #860SPS 
ads.reference_voltage = referenceV
ads.gain = GAIN
ads.data_rate = SAMPLING_RATE
NUM_SAMPLES = 1000

# Single-ended input on channel 0
chan = AnalogIn(ads, ADS.P0)

#Function to collect sample data
def collection(data):
  for _ in range(NUM_SAMPLES):
    raw_value =chan.value
    #voltage = raw_value * 3.3 / 65535  # Convert to voltage (for ADS1115's 16-bit resolution)
    data.append(raw_value) #append collected value to the data collection array
    time.sleep(0.1)  # The delay between each samples
  return data

####### MAIN PROGRAM #######

#### DATA COLLECTION ####
# Collecting data for open hand movement
print("Collecting data for open hand...")
open_hand_data = []
open_hand_data = collection(open_hand_data)

# Delay for user to switch hand movement
print("Please switch hand movement!")
time.sleep(5)

# Collecting data for close hand movement
print("Collecting data for close hand...")
close_hand_data = []
close_hand_data = collection(close_hand_data)


# Plotting and saving the raw data graph for different hand gestures
timer = np.arange(0, NUM_SAMPLES)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(timer, open_hand_data, label='Raw Open Hand Signal')
plt.title('Raw EMG Signal - Open Hand')
plt.xlabel('Samples')
plt.ylabel('Raw EMG values')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(timer, close_hand_data, label='Raw Close Hand Signal')
plt.title('Raw EMG Signal - Close Hand')
plt.xlabel('Samples')
plt.ylabel('Raw EMG values')
plt.legend()
plt.tight_layout()
plt.savefig('Raw_signals.png')
plt.close()



#### DATA FILTERING ####
# Preprocessing: Applying filter to the collected data

# Define bandpass filter parameters
low_cut = 50  # Lower cutoff frequency (slightly lower than the original frequency)
high_cut = 400  # Upper cutoff frequency (slightly higher than the original frequency)
fs = 1000  # Sampling frequency
order = 5  # filter order

# Apply bandpass filter to recover the original signal
def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


filtered_open_hand_data = bandpass_filter(open_hand_data, low_cut, high_cut, fs, order)

filtered_close_hand_data = bandpass_filter(close_hand_data, low_cut, high_cut, fs, order)

# Rectification
filtered_open_hand_signal_rectified = np.abs(filtered_open_hand_data)
filtered_close_hand_signal_rectified = np.abs(filtered_close_hand_data)

# Plotting the original recovered signals and the filtered & rectified signals
plt.figure(figsize=(10, 6))


# Plotting filtered & rectified signals
plt.subplot(2, 2, 1)
plt.plot(np.arange(len(filtered_open_hand_signal_rectified)), filtered_open_hand_signal_rectified , label='Filtered & Rectified Open Hand Signal')
plt.title('Filtered & Rectified Signal - Open Hand')
plt.xlabel('Samples')
plt.ylabel('EMG values/Amplitude')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(np.arange(len(filtered_close_hand_signal_rectified)), filtered_close_hand_signal_rectified, label='Filtered & Rectified Close Hand Signal')
plt.title('Filtered & Rectified Signal - Close Hand')
plt.xlabel('Samples')
plt.ylabel('EMG values/Amplitude')
plt.legend()

plt.tight_layout()
plt.savefig('filtered_rectified_signals.png')
plt.close()


##### FEATURE EXTRACTION #####

def extract_features(chunk_signal):
  average_amplitude = np.mean(chunk_signal) # amp
  rms = np.sqrt(np.mean(chunk_signal**2))   #rms
  slope = np.mean(np.diff(chunk_signal))    #slope
  if np.isnan(slope):                       #exceptional error handling
    slope = 0
  return [average_amplitude, rms, slope]


# Split rectified signals into chunks of 5 samples per chunk
chunk_size = 5
open_hand_chunks = [filtered_open_hand_signal_rectified[i:i+chunk_size] for i in range(0, len(filtered_open_hand_signal_rectified), chunk_size)]

close_hand_chunks = [filtered_close_hand_signal_rectified[i:i+chunk_size] for i in range(0, len(filtered_close_hand_signal_rectified), chunk_size)]

# Extract features from each chunk
open_hand_features = [extract_features(chunk) for chunk in open_hand_chunks]
close_hand_features = [extract_features(chunk) for chunk in close_hand_chunks]

##### DATA NORMALIZATION #####

# Create X vector (feature vectors) and y vector (labels)
X = np.array(open_hand_features + close_hand_features)
y = np.array([0] * len(open_hand_features) + [1] * len(close_hand_features))

print("X:",X)
print("y:",y)

###### TRAINING ######
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set specific C and gamma values
C_value = 1.0
gamma_value = 0.1

# SVM model initialization with RBF kernel and specific C, gamma values
svm_rbf = SVC(kernel='rbf', C=C_value, gamma=gamma_value)

# Training the SVM model
svm_rbf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = svm_rbf.predict(X_test)

# Calculating and printing accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Printing classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Pickling the model for later use
with open('svm_rbf_model.pkl', 'wb') as file:
  pickle.dump(svm_rbf, file)


