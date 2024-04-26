import h5py
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score

from mne.decoding import Scaler, Vectorizer, CSP
from mne.preprocessing import Xdawn
from mne import create_info
from mne import EpochsArray

from mne_features.feature_extraction import FeatureExtractor

import torch
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetBinaryClassifier


data = "PERCEPTION_DATASET_W_PREC_ICA.h5"
with h5py.File(data, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]


sfreq = 512.0
info = create_info(ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 
                                'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 
                                'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 
                                'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 
                                'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 
                                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 
                                'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                                sfreq = sfreq,
                                ch_types='eeg',
                                )



eegnet_parameters = {"chunk_size" : [128],
               "num_electrodes" : [64],
               "dropout" : [0.5],
               "kernel_1" : [64],
               "kernel_2" : [16],
               "F1" : [8],
               "F2" : [16],
               "D" : [2],
               "num_classes" : [2]}


logo = LeaveOneGroupOut()

functions = ["mean", "variance", "std", "pow_freq_bands"]

param_grid = ParameterGrid(eegnet_parameters)

for params in param_grid:

    model = EEGNet(**params)

    #ertmal ohne feature extraction?
    pipeline = Pipeline(
    #Xdawn(n_components=5), #[2,3,4,5,10] reduce noise
    #CSP(), #common spatial pattern, originally for binary classification
    #FeatureExtractor(sfreq=512, selected_funcs=functions, params=None, n_jobs=-1, memory=None),
    [("Scaler", Scaler(info)), #scale the different channels
    #("Vectorizer", Vectorizer()), #to 2D needed as sklearn input but not needed for eegnet input:)
    ("NeuralNet", NeuralNetBinaryClassifier(
        EEGNet,
        max_epochs=10,
        lr=0.001,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ))])



    metrics = {
        "accuracy": (accuracy_score, []),
        "precision": (average_precision_score, []),
        "recall": (recall_score, []),
        "F1-score": (f1_score, [])
    }

    for train_index, test_index in logo.split(X, y, groups=subjects):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        print("X: ", X_train.shape, X_test.shape)
        print("y: ", y_train.shape, y_test.shape)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        for name, (func, scores) in metrics.items():
            score = func(y_test, predictions)
            scores.append(score)
    
        print("scores:", scores)







