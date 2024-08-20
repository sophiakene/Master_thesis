import h5py
import numpy as np
from collections import Counter
from sklearn import svm
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

#reading the principal components
with h5py.File("../visualisations/MODE_PRINCIPAL_COMPS.h5", "r") as f:
    X = f['features'][:]
    y = f['labels'][:]

print(X.shape)
print(y.shape)

###using perception data only (condition 1)
# Convert labels to strings
y_str = y.astype(str)

# Create a mask to filter out the labels ending with '1'
mask = np.array([label.endswith('1') for label in y_str])

# Apply the mask to filter X and y
X_filtered = X[mask]
y_filtered = y[mask]


#convert to mode labels
major_ids = [2, 3, 4, 12, 13, 14, 21, 23, 24]
minor_ids = [1, 11, 22]
mode_labels = []
for l in y_filtered:
    if int(str(l)[:-1]) in minor_ids:
        mode_labels.append(0)
    elif int(str(l)[:-1]) in major_ids:
        mode_labels.append(1)
### 0 = minor, 1 = major
#print(mode_labels[:40])
mode_labels = np.array(mode_labels)
print(mode_labels.shape)
print("shape X_filtered: ", X_filtered.shape)

#leaving out participant 14 so it is not seen during training the svm
#1200 = 20 time windows * 60 trials for one participant
#15 trials for perception data 20*15 = 300
#45 trials for imagination data 20*45 = 900
X_train = X_filtered[:-1200, :] #X[:-1200, :] #X[:, :-60] #all but participant 14
X_test = X_filtered[-1200:, :] #X[-1200:, :] #X[:, -60:] #Participant 14 (last participant chosen as test set -> doing that for everything)
y_train = mode_labels[:-1200] #mode_labels[:-1200]
y_test = mode_labels[-1200:] #mode_labels[-1200:]

# Print the original distribution of the training set
print(f'Original class distribution: {Counter(y_train)}')

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print the new distribution of the training set
#print(f'Resampled class distribution: {Counter(y_train_resampled)}')

# 8-fold cross-validation:
mode_classifier = svm.SVC(kernel="poly", C=0.0001, class_weight='balanced', probability=True)
#mode_classifier.fit(X_train_resampled, y_train_resampled)
mode_classifier.fit(X_train, y_train)
cv = StratifiedKFold(n_splits=8, shuffle=False, random_state=None)
cv_scores_accuracy = cross_val_score(mode_classifier, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
#cv_scores_precision = cross_val_score(mode_classifier, aggregated_data, filtered_labels, cv=cv, scoring='precision_macro')
#cv_scores_recall = cross_val_score(mode_classifier, aggregated_data, filtered_labels, cv=cv, scoring='recall_macro')
#cv_scores_f1 = cross_val_score(mode_classifier, aggregated_data, filtered_labels, cv=cv, scoring='f1_macro')
print("\nACCURACY: ",cv_scores_accuracy)#, 
#"\nPRECISION: ", cv_scores_precision,)
#"\nRECALL: ", cv_scores_recall,
#"\nF1SCORE: ", cv_scores_f1)
y_pred = cross_val_predict(mode_classifier, X_train_resampled, y_train_resampled, cv=cv)
print("here: ", y_test.shape, y_pred.shape)
print(confusion_matrix(y_test, y_pred))


#predicting on test set just to check performance
y_preds = mode_classifier.predict(X_test)
y_pred_proba = mode_classifier.decision_function(X_test)
#print("Accuracy: ", accuracy_score(y_preds, y_test))

# Reshape y_preds to have shape (number of trials, 20 time windows per trial)
num_trials = len(y_test) // 20
y_preds_reshaped = y_preds.reshape(num_trials, 20)

# Apply majority voting to each trial
y_preds_majority_vote = np.array([Counter(trial_preds).most_common(1)[0][0] for trial_preds in y_preds_reshaped])

# Reshape y_test to match the shape of y_preds_majority_vote
y_test_reshaped = y_test.reshape(num_trials, 20)
y_test_majority_vote = np.array([Counter(trial_labels).most_common(1)[0][0] for trial_labels in y_test_reshaped])

# Evaluate the accuracy and AUC and sensitivity
#accuracy = accuracy_score(y_test_majority_vote, y_preds_majority_vote)
print(f'Accuracy: {accuracy_score(y_test_majority_vote, y_preds_majority_vote)}')
print(f'Report:, {classification_report(y_test_majority_vote, y_preds_majority_vote)}')
#print(f'auc: {auc(y_test_majority_vote, y_preds_majority_vote)}')

# Print the predictions and actual labels for the first few trials for inspection
print(f'Predictions: {y_preds_majority_vote}')
print(f'Actual labels: {y_test_majority_vote}')





"""
X = np.mean(X, axis=1) #aggregated into 1 channel
df = pd.DataFrame()

#leaving out participant 14 so it is not seen during training the svm
X_train = X[:-60, :] #X[:, :-60] #all but participant 14
X_test = X[-60:, :] #X[:, -60:] #Participant 14 (last participant chosen as test set -> doing that for everything)
y_train = mode_labels[:-60]
y_test = mode_labels[-60:]
#don't need the test sets here because I'm just training and saving the model

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

mode_classifier = svm.SVC(kernel = "linear", C=0.0001)
mode_classifier.fit(X_train, y_train)
joblib.dump(mode_classifier, 'svm_model_mode.pkl')
print("saved mode_classifier in a pickle file as svm_model_mode.pkl")
"""

