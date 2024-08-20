import h5py
import numpy as np
from collections import Counter
from sklearn import svm
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

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
print(mode_labels.shape) #10800,

# Leaving out participant 14 so it is not seen during training the svm
# 1200 = 20 time windows * 60 trials for one participant
# 15 trials for perception data 20*15 = 300
# 45 trials for imagination data 20*45 = 900
X_train = X_filtered[:-1200, :]
X_test = X_filtered[-1200:, :]
y_train = mode_labels[:-1200]
y_test = mode_labels[-1200:]

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

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

y_pred = model.predict(X_test)
print(y_pred, y_pred.shape) #15 predictions (->trials, not windows) with probabilities for the two classes

joblib.dump(model, 'lstm_model_mode2.pkl')
print("saved mode classifer in a pickle file as lstm_model_mode2.pkl")