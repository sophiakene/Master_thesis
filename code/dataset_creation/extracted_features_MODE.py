import mne
from mne_features.feature_extraction import FeatureExtractor
from scipy.signal import find_peaks
import numpy as np
import h5py
from preprocessing_functions import load_stimuli_metadata
from find_stimulus_length import get_start_and_end
from collections import Counter

import mne
import numpy as np
from preprocessing_functions import load_stimuli_metadata #do i still need this then?
from find_stimulus_length import get_start_and_end
from collections import Counter

participant_ids = ["P01", "P04", "P06", "P07", "P09", "P11", "P12", "P13", "P14"] #excluding P05 as they did
song_ids = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
#condition_ids = [1,2,3,4]

all_epochs = []
all_labels = []
all_subjects = [] 
all_conditions = []
all_stimuli = []


fif_files = ["../../../Thesis/LONGER-EPOCHS-P01-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P04-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P06-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P07-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P09-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P11-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P12-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P13-preprocessed-precICA-raw.fif",
            "../../../Thesis/LONGER-EPOCHS-P14-preprocessed-precICA-raw.fif"]

raws = []
for f in fif_files:
    raw = mne.io.read_raw_fif(f, preload=True)
    raw.del_proj()  # Remove SSP projectors
    raws.append(raw)

# Concatenate raw data
raw_combined = mne.concatenate_raws(raws)

# Find Events
events = mne.find_events(raw_combined)
# change stimulus id to song id
#for event in events:
#    event[2] = str(event[2])[:-1]

included_event_ids = [  11,21,31,41,111,121,131,141,211,221,231,241,
                                12,13,14,
                                22,23,24,
                                32,33,34,
                                42,43,44,
                                112,113,114,
                                122,123,124,
                                132,133,134,
                                142,143,144,
                                212,213,214,
                                222,223,224,
                                232,233,234,
                                242,243,244
                                ]

filtered_events = []
# Iterate through events
for i in range(len(events) - 1):  # Iterate up to the second-to-last event
    current_event = events[i]
    next_event = events[i + 1]
    # Check if the current event is in the included_event_ids
    if current_event[2] in included_event_ids:
        # Check if the next event has event_id = 2000
        if next_event[2] != 2000:
            # If not, include the current event in the filtered events
            filtered_events.append(current_event)
# Check the last event separately to avoid index out of range
last_event = events[-1]
if last_event[2] in included_event_ids:
    filtered_events.append(last_event)


eeg_picks = mne.pick_types(raw_combined.info,
                            meg=False,
                            eeg=True,
                            eog=False,
                            stim=False,
                            include=["FP1", "AF7", "AF3"])

n_windows = 20
window_length = 512 #time points = 1s

params = {"pow_freq_bands__freq_bands": np.array([(0.5, 30)])}

fe = FeatureExtractor(sfreq=512,
    selected_funcs=["rms", 
                    "std", 
                    "skewness",
                    "kurtosis", 
                    "app_entropy", 
                    "hurst_exp",
                    "ptp_amp",  
                    "samp_entropy", 
                    "pow_freq_bands",
                    "line_length"], 
                    n_jobs=-1,
                    params = params)

all_features = []
all_labels = []

for i, event in enumerate(filtered_events):
    if i < 1168:
        participant_id = "P01"
    else:
        participant_id = "P14" 
    event = event.reshape(1, 3)
    stimulus_id = event[0, 2]
    song_id, condition_id = int(str(stimulus_id)[:-1]), int(str(stimulus_id)[-1])
    tmin, tmax = get_start_and_end(song_id, condition_id, participant_id)

    epoch = mne.Epochs(raw_combined, events=event, event_id=stimulus_id, tmin=tmin, tmax=tmax,
                       baseline=(None, None), verbose=False, picks=eeg_picks)
    data = epoch.get_data()[0]  # Shape will be (n_channels, n_samples)

    step_size = (data.shape[1] - window_length) / (n_windows - 1)
    windows = []

    for j in range(n_windows):
        start_idx = int(j * step_size)
        end_idx = start_idx + window_length
        window_data = data[:, start_idx:end_idx]  # Shape (n_channels, window_length)
        windows.append(window_data)

    print(f"Participant: {participant_id}, Number of windows: {len(windows)}, Shape of first window: {windows[0].shape}")
    windows = np.array(windows)

    ### FEATURE EXTRACTION (per window)
    print(len(windows))
    for window in windows:
        features = fe.fit_transform(window[None, :, :])
        all_features.append(features)
        all_labels.append(stimulus_id)

all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

print(f"Features shape: {all_features.shape}")
print(f"Labels shape: {all_labels.shape}")

with h5py.File("MODE_FEATURES.h5", "w") as f:
    f.create_dataset('features', data=all_features)
    f.create_dataset('labels', data=all_labels)

print("dataset created")

    

