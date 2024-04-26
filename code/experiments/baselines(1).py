import warnings
warnings.filterwarnings('ignore')

import h5py
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from mne import create_info
from mne.preprocessing import Xdawn

#create info for the Scaler to use

#bads = []
ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
#chs: 64 EEG
#custom_ref_applied = False
#dig: 67 items (3 Cardinal, 64 EEG)
#file_id: 4 items (dict)
#highpass = 0.5 #Hz
#lowpass = 30.0 #Hz
#meas_date: 2015-03-06 11:30:43 UTC
#meas_id: 4 items (dict)
#nchan = 64
#projs: Average EEG reference: on
sfreq = 512.0 #Hz

info = create_info(ch_names, sfreq, ch_types='eeg', verbose=None)

#data = "dataset_new_all.h5"
#data = "dataset_with_baseline_correction.h5"
#data = "dataset_baseline_correction-200.h5"
#data = "PERCEPTION_DATASET.h5"
#data = "IMAGINATION_DATASET.h5"
#data = "PERCEPTION_DATASET_W_PREC_ICA.h5"

data = "INDIVIDUAL_LENGTH_ALL_CONDITIONS.h5"
#np.set_printoptions(threshold=np.inf) #to see the whole confusion matrix

with h5py.File(data, 'r') as f:
    # List all datasets in the file
    print(f.keys())

    features = f['data'][:]
    subjects = f['subjects'][:]
    labels = f['labels'][:]
    #print(labels)
    conditions = f['condition'][:]
    print("Shape of data vector: ",features.shape) #(2400, 1, 64, 440): 2400 epochs, 1??, 64 channels, 440 time stamps or smth
    #print("HERE: ", features[0,0,:,:])
    print("Shape of subjects vector: ",subjects.shape)
    print("Shape of labels vector: ",labels.shape)
    print("Shape of condition vector: ",conditions.shape)

    #labels = np.array(labels) #might need this?
    labels_to_remove = [1111, 2000, 2001]

    # Find indices of labels to remove
    indices_to_remove = np.where(np.isin(labels, labels_to_remove))[0]
     
    # Remove corresponding EEG data, labels and conditions
    filtered_features = np.delete(features, indices_to_remove, axis=0)
    filtered_labels = np.delete(labels, indices_to_remove)
    filtered_subjects = np.delete(subjects, indices_to_remove)
    filtered_conditions = np.delete(conditions, indices_to_remove)

    
    print("Shape of filtered features: ", filtered_features.shape)
    #aggregate 64 channels into 1 mean channel
    aggregated_data = np.mean(filtered_features, axis=1) #this aggregates all the channels into 1 mean signal. let's not do this now, let's try the vectorizer but still use all of the channels separately
    print("aggegated data shape ", aggregated_data.shape)
    #print("distribution of song labels in data set: ", Counter(filtered_labels))
    """
    print("----------------------------------------------\n")
    print("CLASSIFYING USING SCALER AND VECTORIZER\n")    
    clf = make_pipeline(
    Scaler(info), #but maybe this just scales the different channels? so not really worth it when aggregated?
    #Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
    )

    scores = cross_val_multiscore(clf, filtered_features, filtered_labels, cv=5, n_jobs=None)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    print("Spatio-temporal: %0.1f%%" % (100 * score,))
    
    print("----------------------------------------------")
    """
    #print("WITH BASELINE CORRECTION")

    ###BASELINE song prediction
    #split the data
    
    # Iterate over the splits
    #print("With TimeSplitSeries (n=5):\n")
    # Define the number of splits for TimeSeriesSplit
    #n_splits = 5  # Adjust the number of splits as needed
    # Initialize TimeSeriesSplit
    #tscv = TimeSeriesSplit(n_splits=n_splits)

    #for train_index, test_index in tscv.split(aggregated_data):
    """for train_index, test_index in train_test_split(aggregated_data):
        #print(train_index, test_index, type(train_index), type(test_index))
        X_train, X_test = aggregated_data[train_index], aggregated_data[test_index]
        y_train, y_test = filtered_labels[train_index], filtered_labels[test_index]

        print("distribution of song labels in train set: ", Counter(y_train))
        print("distribution of song labels in test set: ", Counter(y_test))
    
        #X_train, X_test, y_train, y_test = train_test_split(aggregated_data, filtered_labels, test_size=0.2, random_state=42)
        
        #print("elements in training labels: ",sorted(Counter(y_train).items()))
        #print("---------")
        #print("elements in test labels: ", sorted(Counter(y_test).items()))
        #this looks fine, ca. 480 in test data, 1900 in training data per class; each class is represented
        baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
        #baseline_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        #baseline_classifier = DecisionTreeClassifier()
        #baseline_classifier = DummyClassifier(strategy="most_frequent")
        baseline_classifier.fit(X_train, y_train)
        y_pred_baseline = baseline_classifier.predict(X_test)
        print("Baseline evaluation *song prediction* using {}:\n".format(str(baseline_classifier)),
        "Accuracy: ", accuracy_score(y_test, y_pred_baseline),
        "Baseline precision: ", precision_score(y_test, y_pred_baseline, average='macro'), 
        "Baseline recall: ", recall_score(y_test, y_pred_baseline, average='macro'),
        "Baseline f1-score: ", f1_score(y_test, y_pred_baseline, average='macro'),
        "Confusion matrix: ", confusion_matrix(y_test, y_pred_baseline),
        "\n----------------------\n----------------------")
    """
    print("5-FOLD CROSS-VALIDATION OF SONG PREDICTION USING PRECOMPUTED ICA:")
    baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_accuracy = cross_val_score(baseline_classifier, aggregated_data, filtered_labels, cv=cv, scoring='accuracy')
    cv_scores_precision = cross_val_score(baseline_classifier, aggregated_data, filtered_labels, cv=cv, scoring='precision_macro')
    cv_scores_recall = cross_val_score(baseline_classifier, aggregated_data, filtered_labels, cv=cv, scoring='recall_macro')
    cv_scores_f1 = cross_val_score(baseline_classifier, aggregated_data, filtered_labels, cv=cv, scoring='f1_macro')
    print("\nACCURACY: ",cv_scores_accuracy, 
    "\nPRECISION: ", cv_scores_precision,
    "\nRECALL: ", cv_scores_recall,
    "\nF1SCORE: ", cv_scores_f1)
    ###BASELINE meter prediction
    #convert the stimulus labels to binary meter labels

    print("Meter prediction\n")

    three_quarters_id = [1,2,11,12,21,22]
    four_quarters_id = [3,4,13,14,23,24]

    #meter_labels = [3 if label in three_quarters_idx else 4 for label in filtered_labels ]
    meter_labels = []
    for l in filtered_labels:
        if l in three_quarters_id:
            meter_labels.append(0)
        elif l in four_quarters_id:
            meter_labels.append(1)
    ### 0 = 3/4, 1 = 4/4

    """
    clf = make_pipeline(
    #Scaler(info), 
    #Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
    )

    scores = cross_val_multiscore(clf, filtered_features, meter_labels, cv=5, n_jobs=None)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    print("Spatio-temporal: %0.1f%%" % (100 * score,))
    """
    """
    print("with TimeSplitSeries:\n")
    
    for train_index, test_index in tscv.split(aggregated_data):
        #print(train_index, test_index, type(train_index), type(test_index))
        X_train, X_test = aggregated_data[train_index], aggregated_data[test_index]
        y_train, y_test = np.array(meter_labels)[train_index], np.array(meter_labels)[test_index]

        #print(len(meter_labels))
        #X_train, X_test, y_train, y_test = train_test_split(aggregated_data, meter_labels, test_size=0.2, random_state=42)
        baseline_classifier = svm.SVC(kernel="linear", C=0.0001)
        #baseline_classifier = LogisticRegression()
        #baseline_classifier = DummyClassifier(strategy="most_frequent")
        baseline_classifier.fit(X_train, y_train)
        y_pred_baseline = baseline_classifier.predict(X_test)
        print("Baseline Evaluation *meter prediction* using {}:\n".format(str(baseline_classifier)),
        "Accuracy: ", accuracy_score(y_test, y_pred_baseline),
        "Baseline precision: ", precision_score(y_test, y_pred_baseline), 
        "Baseline recall: ", recall_score(y_test, y_pred_baseline), 
        "Baseline f1-score: ", f1_score(y_test, y_pred_baseline),
        "Confusion matrix: ", confusion_matrix(y_test, y_pred_baseline),
        "----------------------\n--------------------")
    """
    print("5-FOLD CROSS-VALIDATION OF METER PREDICTION USING PRECOMPUTED ICA:")
    baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_accuracy = cross_val_score(baseline_classifier, aggregated_data, meter_labels, cv=cv, scoring='accuracy')
    cv_scores_precision = cross_val_score(baseline_classifier, aggregated_data, meter_labels, cv=cv, scoring='precision_macro')
    cv_scores_recall = cross_val_score(baseline_classifier, aggregated_data, meter_labels, cv=cv, scoring='recall_macro')
    cv_scores_f1 = cross_val_score(baseline_classifier, aggregated_data, meter_labels, cv=cv, scoring='f1_macro')
    print("\nACCURACY: ",cv_scores_accuracy, 
    "\nPRECISION: ", cv_scores_precision,
    "\nRECALL: ", cv_scores_recall,
    "\nF1SCORE: ", cv_scores_f1)


    ###BASELINE LYRICS PREDICTION SAME-SONG PAIRS
    #unsure if I should do it on all data points or just the same-song-pairs with and without lyrics?
    
    print("Lyrics prediction:\n")

    #delete instrumental stimuli 21,22,23,24
    instrumental_ids = [21,22,23,24]
    # Find indices of labels to remove
    indices_to_remove = np.where(np.isin(filtered_labels, instrumental_ids))[0]

    filtered_features_wo_instrumental = np.delete(aggregated_data, indices_to_remove, axis=0)
    filtered_labels_wo_instrumental = np.delete(filtered_labels, indices_to_remove)
    filtered_subjects_wo_instrumental = np.delete(filtered_subjects, indices_to_remove)
    #print("here: ", filtered_features_wo_instrumental.shape, filtered_labels_wo_instrumental.shape)
    #convert labels to binary lyrics labels
    
    lyrics_indices = [1,2,3,4]
    non_lyrics_indices = [11,12,13,14]

    lyrics_labels = []
    for l in filtered_labels_wo_instrumental:
        if l in lyrics_indices:
            lyrics_labels.append(1)
        elif l in non_lyrics_indices:
            lyrics_labels.append(0)
    #print(lyrics_labels, len(lyrics_labels))
    #print(filtered_features_wo_instrumental.shape)
    """
    clf = make_pipeline(
    #Scaler(info), 
    #Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
    )

    scores = cross_val_multiscore(clf, filtered_features_wo_instrumental, lyrics_labels,cv=10, n_jobs=None)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    print("Spatio-temporal: %0.1f%%" % (100 * score,))

    """
    lyrics_labels = np.array(lyrics_labels) 
    print("Shape lyrics labels: ", lyrics_labels.shape) 
    print("Shape filtered features wo instrumental: ",filtered_features_wo_instrumental.shape) 
    
    """print("with TimeSplitSeries")
    for train_index, test_index in tscv.split(filtered_features_wo_instrumental):
        X_train, X_test = filtered_features_wo_instrumental[train_index], filtered_features_wo_instrumental[test_index]
        y_train, y_test = lyrics_labels[train_index], lyrics_labels[test_index]
        
        #X_train, X_test, y_train, y_test = train_test_split(filtered_features_wo_instrumental, lyrics_labels, test_size=0.2, random_state=42)
        baseline_classifier = svm.SVC(kernel="linear", C=0.0001)
        #baseline_classifier = LogisticRegression()
        #baseline_classifier = DummyClassifier(strategy="most_frequent")
        baseline_classifier.fit(X_train, y_train)
        y_pred_baseline = baseline_classifier.predict(X_test)
        print("Baseline Evaluation *lyrics prediction* using {}:\n".format(str(baseline_classifier)),
        "Accuracy: ", accuracy_score(y_test, y_pred_baseline),
        "Baseline precision: ", precision_score(y_test, y_pred_baseline), 
        "Baseline recall: ", recall_score(y_test, y_pred_baseline), 
        "Baseline f1-score: ", f1_score(y_test, y_pred_baseline),
        "Confusion matrix: ", confusion_matrix(y_test, y_pred_baseline),
        "----------------------\n--------------------")
    """


    print("5-FOLD CROSS-VALIDATION OF LYRICS PREDICTION USING PRECOMPUTED ICA:")
    baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_accuracy = cross_val_score(baseline_classifier, filtered_features_wo_instrumental, lyrics_labels, cv=cv, scoring='accuracy')
    cv_scores_precision = cross_val_score(baseline_classifier, filtered_features_wo_instrumental, lyrics_labels, cv=cv, scoring='precision_macro')
    cv_scores_recall = cross_val_score(baseline_classifier, filtered_features_wo_instrumental, lyrics_labels, cv=cv, scoring='recall_macro')
    cv_scores_f1 = cross_val_score(baseline_classifier, filtered_features_wo_instrumental, lyrics_labels, cv=cv, scoring='f1_macro')
    print("\nACCURACY: ",cv_scores_accuracy, 
    "\nPRECISION: ", cv_scores_precision,
    "\nRECALL: ", cv_scores_recall,
    "\nF1SCORE: ", cv_scores_f1)








        