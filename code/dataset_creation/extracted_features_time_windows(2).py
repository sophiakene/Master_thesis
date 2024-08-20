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

#extract features for each epoch
#explanation of events and epochs:
#events are marked on the stimulus channel; they show that there is a stimulus onset
#(I think of events as time points)
#further down in the code (l. 113), epochs are created around the events/stimulus onsets.
#(I think of epochs as time spans around events)
def extract_features(epoch):
    peak_features = []
    nle_features = []
    curve_length_features = []
    
    for channel_data in epoch:
        #find_peaks returns a list of the position of the peaks and a list of their magnitudes 
        #and as I am only interested in the number, I only use the position (peak_locs)
        peak_locs, _ = find_peaks(channel_data) 
        #take the length of the list of peak positions to get the number
        number_of_peaks = len(peak_locs) 
        
        nle_value = nonlinear_energy(channel_data)
        curve_length_value = curve_length(channel_data)
        
        peak_features.append(number_of_peaks)
        nle_features.append(nle_value)
        curve_length_features.append(curve_length_value)
    
    #these are the features that are not available in mne_features and therefore have to 
    #be extracted separately. more feature extraction below
    return peak_features, nle_features, curve_length_features

#create empty lists for all the information i want to extract from 
#the preprocessed EEG data
all_epochs = []
all_labels = []
all_subjects = []
all_conditions = []
all_stimuli = []

#for participant_id in participant_ids: #if using my preprocessed data I have to iterate over 
#all participants as they are in separate files. 
#file = "../../../Thesis/LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif".format(participant_id)
#here I'm using Stober's preprocessed data to exclude the possibility of something being wrong with
#my preprocessed data
file = "../../../Thesis/Thesis_Data/OpenMIIR-Perception-512Hz.hdf5"
#read the preprocessed data (it is preprocessed, I just called it raw 
#because no features have been extracted)
raw_data = mne.io.read_raw(file, preload=True)
#finding the events (stimulus onset positions) from the stimulus channel
events = mne.find_events(raw_data, stim_channel="STI 014")
#the stimulus ids are stored in the last column of an event object
#event object: [event_position, None, stimulus_id]
#the stimulus ids consist of the song id (1,2,3,4,11,12,...) and the condition id
#since I am working on the data of the perception condition only, it is always 1
#(so stimulus id 11 means song 1 condition 1 (perception)) and 231 is song 23 condition 1
#(see included_event_ids below in l.79)
stimulus_ids = events[:, -1]
#excluding the channels EXG5+6 because they're not present in all participants' data
#but using all other eeg channels
eeg_picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude=["EXG5", "EXG6"])
#only using perception (of all songs)
included_event_ids = [11, 21, 31, 41, 111, 121, 131, 141, 211, 221, 231, 241]

#only relevant when also using imagination: filtering out trials followed by event id 2000
#(unsuccessful imagination by participant's feedback)
"""filtered_events = []
for i in range(len(events) - 1):
    current_event = events[i]
    next_event = events[i + 1]
    if current_event[2] in included_event_ids and next_event[2] != 2000:
        filtered_events.append(current_event)
last_event = events[-1]
if last_event[2] in included_event_ids:
    filtered_events.append(last_event)

labels = [event[2] for event in filtered_events]"""

filtered_events = events #just renaming events because I am using filtered_events below

#iterate over all events
for event in filtered_events:
    event = event.reshape(1, 3)
    stimulus_id = event[0, 2]
    #split stimulus id into song and condition id
    song_id, condition_id = int(str(stimulus_id)[:-1]), int(str(stimulus_id)[-1])
    #get the start and end time points of a song relative to the stimulus onset
    #the start can be later than the stimulus onset because some conditions (including
    #condition 1, perception) have the cue clicks before the song starts. the end is just the
    #start+length of the song
    #the start and end depend on the song, the condition and the participant as the stimuli
    #slightly changed after the first half of the participants
    #the function is basically a look up table
    tmin, tmax = get_start_and_end(song_id, condition_id, participant_id)
    #according to the start and end time points of the song, epochs are created around the events
    #so each epoch corresponds to one stimulus exposure and is as long as the stimulus (6.8-16s)
    epoch = mne.Epochs(raw_data, events=event, event_id=stimulus_id, tmin=tmin, tmax=tmax,
                        baseline=(None, None), verbose=False, picks=eeg_picks)
    #store epochs etc. in the empty lists
    all_epochs.append(epoch)
    all_conditions.append(condition_id)
    all_labels.append(song_id)
    all_subjects.append(participant_id)
    all_stimuli.append(stimulus_id)

print("len all epochs", len(all_epochs))
print("len all subjects", len(all_subjects))
print("len all labels", len(all_labels))
print("len all conditions", len(all_conditions))

#epoch.get_data() converts the epoch object into a numpy vector
all_epochs = [e.get_data(verbose=False) for e in all_epochs]

#prepare empty list for extracted features
all_windowed_features = []

#settings for sliding windows feature extraction
sfreq = raw_data.info['sfreq'] #512
window_size = 1 #second
n_windows = 20 #as described by niall

#iterate over all epochs
for epoch in all_epochs:
    #get the length of each individual epoch
    #epoch vector = (number of trials, number of channels, number of time points)
    #trials is always 1 here, as I'm taking one epoch at a time
    #number of channels = 64
    #number of time points = length [s] * frequency [1/s] with the frequency being 512 Hz
    epoch_length = epoch.shape[2] / sfreq 
    #the step size depending on the length of the epoch
    step_size = (epoch_length - window_size) / (n_windows - 1)
    
    #one feature list per epoch 
    window_features = []
    #take one window at a time
    for i in range(n_windows):
        start = int(i * step_size * sfreq)
        end = int(start + window_size * sfreq)
        #slize the epoch into windows
        window_data = epoch[:, :, start:end]

        peak_features, nle_features, curve_length_features = extract_features(window_data[0])
        #append the non-mne features as well
        window_features.extend(peak_features)
        window_features.extend(nle_features)
        window_features.extend(curve_length_features)
    
    #append list of features of one epoch to list of all features
    all_windowed_features.append(window_features)

#turn list of features into vector
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
