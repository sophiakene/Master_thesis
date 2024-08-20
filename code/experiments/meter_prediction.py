#just to get the meter predictions
#so i kind of have to save the model
#and then make it make predictions for a specific test set

import h5py
import numpy as np
from collections import Counter
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
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

#convert to meter labels
meter_translation = {1:0,2:0,3:1,4:1,
                11:0,12:0,13:1,14:1,
                21:0,22:0,23:1,24:1}
meter_labels = np.array([meter_translation[label] for label in y]) 
print(meter_labels[:20])

X = np.mean(X, axis=1) #aggregated into 1 channel
# let's try taking the average of T7 and T8 instead -> have to do that in dataset creation (try later)

df = pd.DataFrame()

#X = X.T #transpose X to shape (3519,540)
#print(X.shape)

#leaving out participant 14 so it is not seen during training the svm
X_train = X[:-60, :] #X[:, :-60] #all but participant 14
X_test = X[-60:, :] #X[:, -60:] #Participant 14 (last participant chosen as test set -> doing that for everything)
y_train = meter_labels[:-60]
y_test = meter_labels[-60:]
#don't need the test sets here because I'm just training and saving the model

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

meter_classifier = svm.SVC(kernel = "linear", C=0.0001)
meter_classifier.fit(X_train, y_train)
joblib.dump(meter_classifier, 'svm_model_meter.pkl')
print("saved meter_classifier in a pickle file as svm_model_meter.pkl")

"""#actually testing here to make sure i get the 62%
y_preds = meter_classifier.predict(X_test)
print("Accuracy: ", accuracy_score(y_preds, y_test))


print("8-FOLD CROSS-VALIDATION OF METER PREDICTION")
baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
cv_scores_accuracy = cross_val_score(baseline_classifier, X_train, y_train, cv=cv, scoring='accuracy')
cv_scores_precision = cross_val_score(baseline_classifier, X_train, y_train, cv=cv, scoring='precision_macro')
cv_scores_recall = cross_val_score(baseline_classifier, X_train, y_train, cv=cv, scoring='recall_macro')
cv_scores_f1 = cross_val_score(baseline_classifier, X_train, y_train, cv=cv, scoring='f1_macro')
print("\nACCURACY: ",cv_scores_accuracy, 
"\nPRECISION: ", cv_scores_precision,
"\nRECALL: ", cv_scores_recall,
"\nF1SCORE: ", cv_scores_f1)
y_pred = cross_val_predict(baseline_classifier, X_train, y_train, cv=cv)
print(confusion_matrix(meter_labels, y_pred))"""