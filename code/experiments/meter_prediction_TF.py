import warnings
warnings.filterwarnings('ignore')
from find_stimulus_length import get_start_and_end
import h5py
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, StratifiedKFold
#from sklearn.dummy import DummyClassifier
#from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
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
import mne
from mne import create_info
from mne.preprocessing import Xdawn

#have to take preprocessed data, not dataset :/
all_epochs = []
all_song_ids = []
for participant in ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12','P13', 'P14']:
    file = "../../../Thesis/LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif".format(participant)
    raw_data = mne.io.read_raw_fif(file, preload=True)
    events = mne.find_events(raw_data, stim_channel="STI 014")
    stimulus_ids = events[:, -1]
    #print("stimulus_ids: ", sorted(set(stimulus_ids)))
    eeg_picks = mne.pick_types(raw_data.info, meg=False, eeg=False, eog=False, stim=False, include = ["T8"], exclude=["EXG5", "EXG6"])
    # eeg = True to include all channels
    included_event_ids = [11, 21, 31, 41, 111, 121, 131, 141, 211, 221, 231, 241]  # perception only
    filtered_events = np.array([arr for arr in events if arr[2] in included_event_ids])
    print(filtered_events)
    
    for event in filtered_events:
        print("event before: ", event, event.shape)
        event = event.reshape(1,3)
        print("event after: ", event, event.shape)
        song_id, condition_id = int(str(event[0][2])[:-1]), int(str(event[0][2])[-1])
        tmin, tmax = get_start_and_end(song_id, condition_id, participant)
        rounded_tmin = round(tmin*512) / 512 
        rounded_tmax = round(tmax*512) / 512
        epoch =  mne.Epochs(raw_data, events=event, event_id=event[0][2], tmin=rounded_tmin, 
        tmax=rounded_tmin+6.8709, baseline=(None, None), verbose=False, picks=eeg_picks) #shorten to shortest stimulus but still cutting the cues
        #print("epoch: ", epoch)
        #print(epoch.average())
        print("------------")
        all_epochs.append(epoch.average().get_data().reshape(-1))
        all_song_ids.append(song_id)
        #print(all_epochs)
#print(all_epochs)

print(len(all_epochs)) #540 nice!
print(len(all_song_ids)) #also 540, all good

"""for epoch in all_epochs:
    print(epoch.shape)"""

###meter prediction using time-frequency on one channel, let's see!
#convert the stimulus labels to binary meter labels
print("Meter prediction\n")

three_quarters_id = [1,2,11,12,21,22]
four_quarters_id = [3,4,13,14,23,24]

#meter_labels = [3 if label in three_quarters_idx else 4 for label in filtered_labels ]
meter_labels = []
for l in all_song_ids:
    if l in three_quarters_id:
        meter_labels.append(0)
    elif l in four_quarters_id:
        meter_labels.append(1)
### 0 = 3/4, 1 = 4/4

print("5-FOLD CROSS-VALIDATION OF METER PREDICTION USING TIME-FREQUENCY:")
baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
cv = StratifiedKFold(n_splits=9, shuffle=False)#, random_state=42)
#don't know about the stritified but think this is basically leave one participant out cv bc 9 participants without shuffling
cv_scores_accuracy = cross_val_score(baseline_classifier, all_epochs, meter_labels, cv=cv, scoring='accuracy')
cv_scores_precision = cross_val_score(baseline_classifier, all_epochs, meter_labels, cv=cv, scoring='precision_macro')
cv_scores_recall = cross_val_score(baseline_classifier, all_epochs, meter_labels, cv=cv, scoring='recall_macro')
cv_scores_f1 = cross_val_score(baseline_classifier, all_epochs, meter_labels, cv=cv, scoring='f1_macro')
print("\nACCURACY: ",cv_scores_accuracy, 
"\nPRECISION: ", cv_scores_precision,
"\nRECALL: ", cv_scores_recall,
"\nF1SCORE: ", cv_scores_f1)
y_pred = cross_val_predict(baseline_classifier, all_epochs, meter_labels, cv=cv)
print(confusion_matrix(meter_labels, y_pred))    

