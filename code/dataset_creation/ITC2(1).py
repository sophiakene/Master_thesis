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
                                ]  #ALL


raws = []
for f in fif_files:
    raw = mne.io.read_raw_fif(f, preload=True)
    raw.del_proj()  # Remove SSP projectors
    raws.append(raw)

# Concatenate raw data
raw_combined = mne.concatenate_raws(raws)

# Find Events
events = mne.find_events(raw_combined)

#filter out unsuccessful imaginations
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

filtered_events2 = []
for event in filtered_events:
    if event[2] < 1000:
        filtered_events2.append(event)

#perception condition only:
filtered_events3 = []
for event in filtered_events2:
    if str(event[2])[-1] == "1":
        filtered_events3.append(event)

print(len(filtered_events3))


events = np.array(filtered_events3)

# change stimulus id to song id
#change stimulus id to meter label 
stim_to_meter = {"1":0, "2":0, "3":1, "4":1, 
                "11":0, "12":0, "13":1, "14":1,
                "21":0, "22":0, "23":1, "24":1}

for event in filtered_events3:
    stim_id = str(event[2])[:-1]
    event[2] = stim_to_meter[stim_id]
    #print(stim_id, event[2])

print(len(filtered_events3))

print(filtered_events3[:5])
print(filtered_events3[0][2])
print(type(filtered_events3[0][2])) #np int

epochs3_4 = mne.Epochs(raw_combined, 
                    filtered_events3, 
                    event_id = 0,
                    tmin=-0.2,
                    tmax=9.5,
                    baseline=(-0.2, 0),
                    preload=True)
                
print(len(epochs3_4))

freqs = np.arange(1, 30, 1)
n_cycles = freqs / 2.0

power3_4, itc3_4 = mne.time_frequency.tfr_morlet(epochs3_4, 
                                            freqs=freqs,
                                            n_cycles=n_cycles,
                                            return_itc=True)


itc_data3_4 = itc3_4.data

print(itc_data3_4)


#np.save("itc_values_3_4meter.npy", itc_data3_4)

epochs4_4 = mne.Epochs(raw_combined, 
                    filtered_events3, 
                    event_id = 1,
                    tmin=-0.2,
                    tmax=9.5,
                    baseline=(-0.2, 0),
                    preload=True)
                
print(len(epochs4_4))

freqs = np.arange(1, 30, 1)
n_cycles = freqs / 2.0

power4_4, itc4_4 = mne.time_frequency.tfr_morlet(epochs4_4, 
                                            freqs=freqs,
                                            n_cycles=n_cycles,
                                            return_itc=True)


itc_data4_4 = itc4_4.data
#time_window = (0, 6.87)
#freq_band = (12,30) #beta

print(itc_data4_4)

np.save("itc_values_4_4meter.npy", itc_data4_4)

print(itc_data3_4[0].shape)

"""
# Step 1: Find the 15 highest values and their positions for each of the 29 rows
top_15_rows = []
for row in itc_data3_4:
    indices = np.argpartition(row, -15)[-15:]  # Get indices of 15 largest elements
    top_15_values = row[indices]  # Get the values of these indices
    sorted_indices = indices[np.argsort(-top_15_values)]  # Sort indices by the actual values in descending order
    top_15_rows.append((top_15_values[np.argsort(-top_15_values)], sorted_indices))

# Step 2: Find the 50 highest values and their positions for each of the 4967 columns
top_50_columns = []
for col in itc_data3_4.T:
    indices = np.argpartition(col, -50)[-50:]  # Get indices of 50 largest elements
    top_50_values = col[indices]  # Get the values of these indices
    sorted_indices = indices[np.argsort(-top_50_values)]  # Sort indices by the actual values in descending order
    top_50_columns.append((top_50_values[np.argsort(-top_50_values)], sorted_indices))

print("3/4 top 15 frequencies: ", top_15_rows)
print("3/4 top 50 time points: ", top_15_rows)

#########
# Step 1: Find the 15 highest values and their positions for each of the 29 rows
top_15_rows = []
for row in itc_data4_4:
    indices = np.argpartition(row, -15)[-15:]  # Get indices of 15 largest elements
    top_15_values = row[indices]  # Get the values of these indices
    sorted_indices = indices[np.argsort(-top_15_values)]  # Sort indices by the actual values in descending order
    top_15_rows.append((top_15_values[np.argsort(-top_15_values)], sorted_indices))

# Step 2: Find the 50 highest values and their positions for each of the 4967 columns
top_50_columns = []
for col in itc_data4_4.T:
    indices = np.argpartition(col, -50)[-50:]  # Get indices of 50 largest elements
    top_50_values = col[indices]  # Get the values of these indices
    sorted_indices = indices[np.argsort(-top_50_values)]  # Sort indices by the actual values in descending order
    top_50_columns.append((top_50_values[np.argsort(-top_50_values)], sorted_indices))

print("4/4 top 15 frequencies: ", top_15_rows)
print("4/4 top 50 time points: ", top_15_rows)
"""