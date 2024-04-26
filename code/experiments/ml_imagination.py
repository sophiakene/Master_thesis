import h5py
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from mne.decoding import Scaler, Vectorizer, CSP
from mne.preprocessing import Xdawn
from mne_features.feature_extraction import FeatureExtractor
from mne import create_info
from mne import EpochsArray
import torch
from torcheeg.models import EEGNet
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold


data = "IMAGINATION_DATASET.h5"
data = "NEW_IMAGINATION_ICA-2000.h5"

with h5py.File(data, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

aggregated_data = np.mean(X, axis=1)

print("5-FOLD CROSS-VALIDATION OF SONG PREDICTION ON IMAGINATION DATA:")
baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_accuracy = cross_val_score(baseline_classifier, aggregated_data, y, cv=cv, scoring='accuracy')
cv_scores_precision = cross_val_score(baseline_classifier, aggregated_data, y, cv=cv, scoring='precision_macro')
cv_scores_recall = cross_val_score(baseline_classifier, aggregated_data, y, cv=cv, scoring='recall_macro')
cv_scores_f1 = cross_val_score(baseline_classifier, aggregated_data, y, cv=cv, scoring='f1_macro')

print("\nACCURACY: ",cv_scores_accuracy, 
"\nPRECISION: ", cv_scores_precision,
"\nRECALL: ", cv_scores_recall,
"\nF1SCORE: ", cv_scores_f1)


