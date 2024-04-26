import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
from sklearn import svm

perception_data = "PERCEPTION_DATASET_W_PREC_ICA.h5"

with h5py.File(perception_data, 'r') as f1:
    # List all datasets in the file
    print(f1.keys())
    X = f1['data'][:]
    subjects = f1['subjects'][:]
    y = f1['labels'][:]
    conditions = f1['condition'][:]

X = np.mean(X, axis=1)
#X = X.ravel()
print(X.shape)

# Sample EEG features DataFrame (replace this with your actual data)
# Assuming eeg_features contains features extracted from EEG data for each stimulus
eeg_features = pd.DataFrame(X)  # Your EEG features data #This is raw data now, just averaged into 1 channel...

# Sample ground truth labels for each classification subtask
meter_translation = {1:'3/4',2:'3/4',3:'4/4',4:'4/4',
                11:'3/4',12:'3/4',13:'4/4',14:'4/4',
                21:'3/4',22:'3/4',23:'4/4',24:'4/4'}
lyrics_translation = {1:'with',2:'with',3:'with',4:'with',
                11:'without',12:'without',13:'without',14:'without',
                21:'without',22:'without',23:'without',24:'without'}
mode_translation = {1: 'minor', 2: 'major', 3: 'major', 4: 'major', 
                    11: 'minor', 12: 'major', 13: 'major', 14: 'major',
                    21: 'major', 22: 'minor', 23: 'major', 24: 'major'}


meter_labels = np.array([meter_translation[label] for label in y]) 
lyrics_labels = np.array([lyrics_translation[label] for label in y])
mode_labels = np.array([mode_translation[label] for label in y]) 


print(meter_labels.shape)
# Combine ground truth labels into a single DataFrame
# Each row corresponds to a stimulus, and columns represent the labels of each subtask
GT_df = pd.DataFrame({
    'Meter': meter_labels,
    'Lyrics': lyrics_labels,
    'Mode': mode_labels
})

X = np.concatenate((X, meter_labels, lyrics_labels, mode_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
baseline_classifier.fit(X_train, y_train)
y_preds = baseline_classifier.predict(X_test)
print(y_preds)
print(accuracy_score(y_preds, y_test))