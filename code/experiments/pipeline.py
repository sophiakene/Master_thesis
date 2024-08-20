import h5py
import numpy as np
from sklearn import svm
import pandas as pd
import joblib

# Load the EEG data from the HDF5 file
data_path = "../dataset_creation/TRIMMED_PERCEPTION.h5"
# data_path = "../dataset_creation/TRIMMED_ALL_CONDITIONS.h5"
# data_path = "../dataset_creation/TRIMMED_IMAGINATION.h5"
with h5py.File(data_path, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

# Channel average
X = np.mean(X, axis=1)  # Shape: (540, 3519)

# Split into train and test data (P14 = test)
X_train = X[:-60, :]  # Shape: (480, 3519)
y_train = y[:-60]     # Shape: (480,)
X_test = X[-60:, :]   # Shape: (60, 3519)
y_test = y[-60:]      # Shape: (60,)

# Load the pre-trained classifiers
meter_classifier = joblib.load("svm_model_meter.pkl")
mode_classifier = joblib.load("svm_model_mode.pkl")
lyrics_classifier = joblib.load("svm_model_lyrics.pkl")

# Making predictions on participant 14's data
meter_predictions = meter_classifier.predict(X_test)
mode_predictions = mode_classifier.predict(X_test)
lyrics_predictions = lyrics_classifier.predict(X_test)

# Create a dictionary with the predictions
data = {
    'Meter_pred': meter_predictions,
    'Mode_pred': mode_predictions,
    'Lyrics_pred': lyrics_predictions
}

# Transpose X_test to shape (3519, 60) to match trials with the correct format
X_test = X_test.T

# Add EEG data to the dictionary
for i in range(X_test.shape[0]):
    data[f'time_point_{i+1}'] = X_test[i]

# Create DataFrame using pd.concat for better performance
df = pd.concat([pd.DataFrame(data)], axis=1)

print(df.head())
