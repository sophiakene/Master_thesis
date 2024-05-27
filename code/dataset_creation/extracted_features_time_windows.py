import mne
from mne_features.feature_extraction import FeatureExtractor
from scipy.signal import find_peaks
import numpy as np
import h5py
from preprocessing_functions import load_stimuli_metadata
from find_stimulus_length import get_start_and_end
from collections import Counter

participant_ids = ["P01", "P04", "P06", "P07", "P09", "P11", "P12", "P13", "P14"] #not using P05

#defining functions for features not available in mne_features
def nonlinear_energy(signal):
    nle = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    return np.mean(nle)

def curve_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def extract_features(epoch):
    peak_features = []
    nle_features = []
    curve_length_features = []
    
    for channel_data in epoch:
        peak_locs, _ = find_peaks(channel_data)
        number_of_peaks = len(peak_locs)
        
        nle_value = nonlinear_energy(channel_data)
        curve_length_value = curve_length(channel_data)
        
        peak_features.append(number_of_peaks)
        nle_features.append(nle_value)
        curve_length_features.append(curve_length_value)
    
    return peak_features, nle_features, curve_length_features

all_epochs = []
all_labels = []
all_subjects = []
all_conditions = []
all_stimuli = []

with open("num_filtered_labels.txt", "w") as f:
    for participant_id in participant_ids:
        file = "../../../Thesis/LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif".format(participant_id)
        file = "../../../Thesis/Thesis_Data/OpenMIIR-Perception-512Hz.hdf5"
        raw_data = mne.io.read_raw(file, preload=True)
        events = mne.find_events(raw_data, stim_channel="STI 014")
        stimulus_ids = events[:, -1]
        #excluding EXG5+6 because they're not present in all participants' data
        eeg_picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude=["EXG5", "EXG6"])
        #only using perception (of all songs)
        included_event_ids = [11, 21, 31, 41, 111, 121, 131, 141, 211, 221, 231, 241]

        #only relevant when also using imagination: filtering out trials followed by event id 2000
        #(unsuccessful imagination by participant's feedback)
        filtered_events = []
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            if current_event[2] in included_event_ids and next_event[2] != 2000:
                filtered_events.append(current_event)
        last_event = events[-1]
        if last_event[2] in included_event_ids:
            filtered_events.append(last_event)

        labels = [event[2] for event in filtered_events]


        for event in filtered_events:
            event = event.reshape(1, 3)
            stimulus_id = event[0, 2]
            song_id, condition_id = int(str(stimulus_id)[:-1]), int(str(stimulus_id)[-1])
            tmin, tmax = get_start_and_end(song_id, condition_id, participant_id)

            epoch = mne.Epochs(raw_data, events=event, event_id=stimulus_id, tmin=tmin, tmax=tmax,
                               baseline=(None, None), verbose=False, picks=eeg_picks)
            all_epochs.append(epoch)
            all_conditions.append(condition_id)
            all_labels.append(song_id)
            all_subjects.append(participant_id)
            all_stimuli.append(stimulus_id)

print("len all epochs", len(all_epochs))
print("len all subjects", len(all_subjects))
print("len all labels", len(all_labels))
print("len all conditions", len(all_conditions))

all_epochs = [e.get_data(verbose=False) for e in all_epochs]

all_windowed_features = []

sfreq = raw_data.info['sfreq'] #512
window_size = 1
n_windows = 20

for epoch in all_epochs:
    epoch_length = epoch.shape[2] / sfreq 
    step_size = (epoch_length - window_size) / (n_windows - 1)
    
    window_features = []
    for i in range(n_windows):
        start = int(i * step_size * sfreq)
        end = int(start + window_size * sfreq)
        window_data = epoch[:, :, start:end]

        peak_features, nle_features, curve_length_features = extract_features(window_data[0])
        window_features.extend(peak_features)
        window_features.extend(nle_features)
        window_features.extend(curve_length_features)
    
    all_windowed_features.append(window_features)

all_windowed_features = np.array(all_windowed_features)

# Feature extraction for each epoch
fe = FeatureExtractor(sfreq=sfreq, selected_funcs=["skewness", "mean", "kurtosis", "app_entropy", "hurst_exp"], n_jobs=-1)

# Transform each epoch separately
X = []
for epoch in all_epochs:
    X.append(fe.fit_transform(epoch))

X = np.vstack(X)  # put into np vector vertically 

# Combine the features horizontally
all_features = np.hstack([all_windowed_features, X])

all_labels = np.array(all_labels)
all_subjects = [s.encode("ascii", "ignore") for s in all_subjects]
all_conditions = np.array(all_conditions)

with h5py.File("FEATURES_DATASET_PERCEPTION_STOBER_PREC.h5", "w") as f: #this took 2.5h with 32 cores
    f.create_dataset('data', data=all_features)
    f.create_dataset('labels', data=all_labels)
    f.create_dataset('subjects', data=all_subjects)
    f.create_dataset('condition', data=all_conditions)

print("dataset created")
