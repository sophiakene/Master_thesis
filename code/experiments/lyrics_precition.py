#on all stimuli; don't distinguish between same song and different song pairs
import h5py
import numpy as np
from collections import Counter
from sklearn import svm
import pandas as pd
import joblib

# Load the EEG data from the HDF5 file
data_path = "../dataset_creation/TRIMMED_PERCEPTION.h5"
#data_path = "../dataset_creation/TRIMMED_ALL_CONDITIONS.h5"
#data_path = "../dataset_creation/TRIMMED_IMAGINATION.h5"
with h5py.File(data_path, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

print(X.shape)  # (540, 64, 3538) actually 3519 in this dataset
print(y.shape)  # (540,)
print(Counter(y))  # 45 per each of the 12 classes

#convert to mode labels
lyrics_ids = [1,2,3,4]
non_lyrics_ids = [11,12,13,14,21,22,23,24]

lyrics_labels = []
for l in y:
    if l in lyrics_ids:
        lyrics_labels.append(1)
    elif l in non_lyrics_ids:
        lyrics_labels.append(0)
### 0 = minor, 1 = major

X = np.mean(X, axis=1) #aggregated into 1 channel
df = pd.DataFrame()

#leaving out participant 14 so it is not seen during training the svm
X_train = X[:-60, :] #X[:, :-60] #all but participant 14
X_test = X[-60:, :] #X[:, -60:] #Participant 14 (last participant chosen as test set -> doing that for everything)
y_train = lyrics_labels[:-60]
y_test = lyrics_labels[-60:]
#don't need the test sets here because I'm just training and saving the model

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

lyrics_classifier = svm.SVC(kernel = "linear", C=0.0001)
lyrics_classifier.fit(X_train, y_train)
joblib.dump(lyrics_classifier, 'svm_model_lyrics.pkl')
print("saved lyrics_classifier in a pickle file as svm_model_lyrics.pkl")
