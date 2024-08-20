import mne
from mne_features.feature_extraction import FeatureExtractor
from mne.preprocessing import peak_finder
from scipy.signal import find_peaks

import numpy as np
import h5py

from preprocessing_functions import load_stimuli_metadata #do i still need this then?
from find_stimulus_length import get_start_and_end

from collections import Counter

participant_ids = ["P01", "P04", "P06", "P07", "P09", "P11", "P12", "P13", "P14"] #excluding P05 as they did

path = "openmiir/eeg/preprocessing/notebooks"

all_epochs = []
all_labels = []
all_subjects = [] 
all_conditions = []
all_stimuli = []

for participant_id in participant_ids:
    #-1,16s epochs 
    file = "../../../Thesis/LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif".format(participant_id)
    raw_data = mne.io.read_raw_fif(file, preload = True)
    events = mne.find_events(raw_data, stim_channel="STI 014")
    print("number of events: ", len(events))
    #triplets of time, - , stimulus id
    stimulus_ids = events[:, -1]
    print("stimulus_ids:", stimulus_ids)
    print(Counter(stimulus_ids))

    eeg_picks = mne.pick_types(raw_data.info, include=["T8"])
    included_event_ids = [11,21,31,41,111,121,131,141,211,221,231,241] #perception
                
    #filter the events. in imagination have to exclude trials that are followed by 2000
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

    print("len events", len(events))
    print("len filtered events", len(filtered_events))
    labels = [event[2] for event in filtered_events]
    print(Counter(labels))

    for event in filtered_events:
        event = event.reshape(1,3)
        stimulus_id = event[0,2]
        #print(stimulus_id, str(stimulus_id)[-1])
        
        #if stimulus_id <1000 and str(stimulus_id)[-1] not in ["3","4"]: #just perception events
        #print(stimulus_id)
        song_id, condition_id = int(str(stimulus_id)[:-1]), int(str(stimulus_id)[-1]) #splitting stimulus id into song and condition
        #print(stimulus_id, song_id, condition_id) #correct
        tmin, tmax = get_start_and_end(song_id, condition_id, participant_id)
        print("Length of song = ", tmax-tmin)
        #print(stimulus_id)

        #print("tmin: ", tmin, "\ntmax: ", tmax)

        #INDIVIDUAL LENGTHS
        epoch = mne.Epochs(raw_data, events=event, event_id=stimulus_id, tmin=tmin, tmax=tmin+6.8709, #tmax=6.8709
        baseline=(None,None), verbose=False, picks=eeg_picks)
        print("EPOCH ", epoch.get_data()[0].shape)
        all_epochs.append(epoch)
        #print("should increase by 1: ", len(all_epochs))
        all_conditions.append(condition_id)
        all_labels.append(song_id)
        all_subjects.append(participant_id)
        all_stimuli.append(stimulus_id)

print("len all epochs", len(all_epochs))
print("len all subjects", len(all_subjects))
print("len all labels", len(all_labels))
print("len all conditions", len(all_conditions))

#all_epochs = np.array([e.get_data(verbose=False) for e in all_epochs])

#HERE FEATURE EXTRACTION INSTEAD OF TRIMMING / PADDING
#nvm, have to trim or pad (doing trimming now, bc that makes them smaller)

all_epochs = [e.get_data(verbose=False) for e in all_epochs]

new_all_epochs = []
for epoch in all_epochs:
    epoch = epoch[:,:,:3518]
    new_all_epochs.append(epoch)

all_epochs = np.array(new_all_epochs)

print("before squeezing: ", all_epochs[0].shape)

all_epochs = np.squeeze(all_epochs, axis=1)

print("after squeezing: ",all_epochs[0].shape)
print(all_epochs.shape)


fe = FeatureExtractor(sfreq = 512, selected_funcs=["app_entropy"],n_jobs=-1)
X = fe.fit_transform(all_epochs) #this is a numpy array

#print(all_epochs.shape)
print(X.shape) #1 mean value for each of the 540 trials -> all participants
#why is it just 5 per stimulus?!
print(len(all_labels))
print(X[0:10])
print(all_labels[0:10]) 

print("--------------")

z = zip(X, all_labels)

dic = {}
for x,l in z:
    print(l,": ",x)
    if l in dic:
        dic[l].append(x)
    else:
        dic[l] = [x]

print(len(dic)) #12, one for each stimulus

print(dic)







