import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import h5py
import numpy as np
from sklearn import svm
from mne_features.feature_extraction import FeatureExtractor
from collections import Counter
import seaborn as sns

perception_data = "../../../Thesis/PERCEPTION_DATASET_W_PREC_ICA.h5"

import os
print(os.getcwd())

with h5py.File(perception_data, 'r') as f1:
    # List all datasets in the file
    print(f1.keys())
    X = f1['data'][:]
    subjects = f1['subjects'][:]
    y = f1['labels'][:]
    conditions = f1['condition'][:]

print(X.shape)
X = np.mean(X, axis=1) #don't aggregate over channels when doing feature extraction
#X = X.ravel()
print(X.shape)

# Sample EEG features DataFrame (replace this with your actual data)
# Assuming eeg_features contains features extracted from EEG data for each stimulus
#eeg_features = pd.DataFrame(X)  # Your EEG features data #This is raw data now, just averaged into 1 channel...

# Sample ground truth labels for each classification subtask
meter_translation = {1:0,2:0,3:1,4:1,
                11:0,12:0,13:1,14:1,
                21:0,22:0,23:1,24:1}
lyrics_translation = {1:1,2:1,3:1,4:1,
                11:0,12:0,13:0,14:0,
                21:0,22:0,23:0,24:0}
mode_translation = {1: 0, 2: 1, 3: 1, 4: 1, 
                    11: 0, 12: 1, 13: 1, 14: 1,
                    21: 1, 22: 0, 23: 1, 24: 1}


meter_labels = np.array([meter_translation[label] for label in y]) 
lyrics_labels = np.array([lyrics_translation[label] for label in y])
mode_labels = np.array([mode_translation[label] for label in y]) 


print(meter_labels.shape)
# Combine ground truth labels into a single DataFrame
# Each row corresponds to a stimulus, and columns represent the labels of each subtask
GT_df = pd.DataFrame({
    'Meter': meter_labels,
    'Lyrics': lyrics_labels,
    'Mode': mode_labels,

})
print(GT_df[:10])
print(len(GT_df))

X = X.T #transpose X to shape (3538,540)
for i in range(X.shape[0]):
    GT_df[f'time_point{i+1}'] = X[i]

print(GT_df[:10])
print(len(GT_df))

#X = np.concatenate((X, meter_labels, lyrics_labels, mode_labels))
#X is the EEG data 

### FEATURE EXTRACTION (in this case don't use channel average)
"""fe = FeatureExtractor(selected_funcs=['mean', 'pow_freq_bands']) #just two features to test:)
X = fe.fit_transform(X)"""


# Split the data into training and testing sets
# replace with doing leave-one-participant-out cv
X_train, X_test, y_train, y_test = train_test_split(GT_df, y, test_size=0.2, random_state=42, shuffle=False) #maybe stratified as the classes aren't that uniformly distributed...
print("y_train: ", Counter(y_train))


baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
#scores = cross_val_score(baseline_classifier, X, y, cv=9) #apparently shuffle=False
baseline_classifier.fit(X_train, y_train)
y_preds = baseline_classifier.predict(X_test)
print(y_preds)
print(accuracy_score(y_preds, y_test))

#print([(p,t) for (p,t) in zip(y_preds, y_test)])
cf_matrix = confusion_matrix(y_test, y_preds)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
