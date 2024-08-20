import h5py
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Reading the principal components
with h5py.File("../visualisations/MODE_PRINCIPAL_COMPS.h5", "r") as f:
    X = f['features'][:]
    y = f['labels'][:]

print(X.shape)
print(y.shape)

# Using perception data only (condition 1)
# Convert labels to strings
y_str = y.astype(str)

# Create a mask to filter out the labels ending with '1'
mask = np.array([label.endswith('1') for label in y_str])

# Apply the mask to filter X and y
X_filtered = X[mask]
y_filtered = y[mask]

# Convert to mode labels
major_ids = [2, 3, 4, 12, 13, 14, 21, 23, 24]
minor_ids = [1, 11, 22]
mode_labels = []
for l in y_filtered:
    if int(str(l)[:-1]) in minor_ids:
        mode_labels.append(0)
    elif int(str(l)[:-1]) in major_ids:
        mode_labels.append(1)

# 0 = minor, 1 = major
mode_labels = np.array(mode_labels)
print(mode_labels.shape)

# Leaving out participant 14 so it is not seen during training the svm
# 1200 = 20 time windows * 60 trials for one participant
# 15 trials for perception data 20*15 = 300
# 45 trials for imagination data 20*45 = 900
X_train = X_filtered[:-300, :]
X_test = X_filtered[-300:, :]
y_train = mode_labels[:-300]
y_test = mode_labels[-300:]

# Print the original distribution of the training set
print(f'Original class distribution: {Counter(y_train)}')

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(X_train_resampled.shape)

# Reshape the input data to 3D shape expected by LSTM layers
# Assume each sample should be reshaped to (20, 5) as there are 20 time windows and 5 features
n_trials_train = X_train_resampled.shape[0] // 20
n_trials_test = X_test.shape[0] // 20

X_train_resampled = X_train_resampled.reshape((n_trials_train, 20, 5))
X_test = X_test.reshape((n_trials_test, 20, 5))

# One-hot encode the labels
y_train_resampled = pd.get_dummies(y_train_resampled).values.reshape((n_trials_train, 20, -1))[:, 0, :]
y_test = pd.get_dummies(y_test).values.reshape((n_trials_test, 20, -1))[:, 0, :]

# Define the model
model = Sequential()
model.add(LSTM(256, input_shape=(20, 5), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Add Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Define a custom callback for validation with majority voting
class ValidationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred_windows = self.model.predict(self.validation_data[0])
        y_pred_labels = np.argmax(y_pred_windows, axis=1)
        
        y_pred_trials = []
        for i in range(n_trials_test):
            trial_predictions = y_pred_labels[i*20:(i+1)*20]
            majority_vote = Counter(trial_predictions).most_common(1)[0][0]
            y_pred_trials.append(majority_vote)

        y_val_labels = np.argmax(self.validation_data[1], axis=1)
        trial_accuracy = accuracy_score(y_val_labels, y_pred_trials)
        print(f'\nEpoch {epoch + 1}: Trial-based Validation Accuracy: {trial_accuracy}')

validation_callback = ValidationCallback()

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping, validation_callback], verbose=1)

# Make predictions for each time window in the test set
y_pred_windows = model.predict(X_test)

# Convert the predictions from one-hot encoding back to class labels
y_pred_labels = np.argmax(y_pred_windows, axis=1)

# Perform majority voting for each trial
y_pred_trials = []
for i in range(n_trials_test):
    trial_predictions = y_pred_labels[i*20:(i+1)*20]
    majority_vote = Counter(trial_predictions).most_common(1)[0][0]
    y_pred_trials.append(majority_vote)

# Convert one-hot encoded test labels back to class labels for comparison
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate the predictions for the full trials
trial_accuracy = accuracy_score(y_test_labels, y_pred_trials)
print(f'Trial-based Test Accuracy: {trial_accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test_labels, y_pred_trials))
print('Classification Report:')
print(classification_report(y_test_labels, y_pred_trials))
