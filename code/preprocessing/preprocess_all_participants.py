# Preprocessing all raw data files in a loop, writing preprocessed data to 1 fif file per participant

import mne 
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
import numpy as np
#import matplotlib.pyplot as plt
import os
import xlrd
from preprocessing_functions import *
#from mne.preprocessing import EOGRegression

participants = ['P01', 'P04', 'P05', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
participants = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14'] #excluding P05, don't have ica data for them

folder = 'openmiir/raw_data'


EOG_channels = ['EXG'+str(n)for n in range(1,7)]

for participant in participants:
    file_path = os.path.join(folder, '{}-raw.fif'.format(participant))
    raw = mne.io.read_raw_fif(file_path, preload = True)

    bad_channels = raw.info["bads"] #+ ["EXG5", "EXG6"]
    print("bad channels {}:".format(participant), bad_channels, "\n\n")

    print("all channels {}".format(participant), raw.info["ch_names"])

    if "EXG5" in raw.info["ch_names"] and "EXG6" in raw.info["ch_names"]:
        raw.drop_channels(["EXG5", "EXG6"])

    if bad_channels: #only if there are any
        # Load a standard montage to get sensor locations
        montage = make_standard_montage('biosemi64')
        # Apply the montage to raw data
        raw.set_montage(montage, on_missing='ignore')
        # Interpolate bad channels
        #raw.interpolate_bads(exclude=EOG_channels)
        raw.interpolate_bads() #for older version


    merge_trial_and_audio_onsets(raw, verbose = False)


    # bandpass filter
    raw.filter(0.5, 30, filter_length='auto', #changed filter length from 10s to auto bc it didn't work
                l_trans_bandwidth=0.1, h_trans_bandwidth=0.5,
                method='fft', iir_params=None,
                picks=None, n_jobs=1, verbose=False)
    
    trial_events = mne.find_events(raw)
    
    # generate events on the beats
    generate_beat_events(trial_events,                  # base events as stored in raw fif files
                         sr=512.0,                      # sample rate, correct value important to compute event frames
                         verbose=False,)
    
    # create epochs around the events in the eye channels to remove them later on during 
    eog_event_id = 5000 #index assigned to the found events
    eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id, verbose=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=True, exclude=[]) # FIXME
    tmin = -1 #-.5
    tmax = 16 #6.8709 #.5
    eog_epochs = mne.Epochs(raw, events=eog_events, event_id=eog_event_id,
                        tmin=tmin, tmax=tmax, proj=False, picks=picks,
                        preload=True, verbose=False)

    print("Number of EOG epochs: ", len(eog_epochs))
    
    # downsample the raw data
    # in the perception experiments stober et al did, they did not downsample but left the data at 512Hz
    #raw.resample(64, verbose=True)


    #using EOG artifact removal instead of ICA now
    #also do one attempt with ICA not [:100] but a little more and like randomly drawn? not first 100
    """raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2', 'EXG3', 'EXG4'])
    weights = EOGRegression().fit(raw)
    raw_clean = weights.apply(raw, copy=True)"""

    
    # independent component analysis to get rid of eye blinking data
    """reject_dict = {
    "P01" : [0,1,3,11], 
    "P04" : [0,2],
    "P05" : [1, 2, 13, 18, 33, 36, 62],
    "P06" : [0, 2, 11, 18, 38],
    "P07" : [0, 1, 2, 13, 21, 39],
    "P09" : [0, 1, 3, 6, 8, 18],
    "P11" : [0, 3, 17, 30, 47, 60],
    "P12" : [0, 1, 3, 13, 23, 38, 60], 
    "P13" : [0, 1, 2, 3, 5, 13, 24, 46, 58],
    "P14" : [0, 1, 2, 4, 8, 11]
    }
    print("Doing ICA for {} now".format(participant))
    ica = mne.preprocessing.ICA(n_components=64, method="infomax", fit_params=dict(extended=True),random_state=42, verbose=False)
    ica.fit(eog_epochs[0:100])
    print("plotting now")
    ica.plot_components()
    print("plotted")
    ica.exclude = reject_dict[participant]

    cleaned_raw = ica.apply(raw)
    cleaned_raw.save('NEWEST-{}-preprocessed-ica.fif'.format(participant), overwrite=True) 
    raw.plot()
    cleaned_raw.plot()"""
    # ica is applied in the new raw?


    ica = mne.preprocessing.read_ica('{}-100p_64c-ica.fif'.format(participant),verbose=False)
    print("***********",ica.n_components_)
    #ica.n_components_ = 0.99
    raw_clean = ica.apply(raw)
    raw_clean.save('LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif'.format(participant), overwrite=True)


    """raw_clean.save('NEW-EOGartifactremoval-{}-preprocessed-raw.fif'.format(participant), overwrite=True)
    """
"""
# Open an HDF5 file to store the dataset
with h5py.File('eeg_dataset.h5', 'w') as hf:
    # Iterate through each participant's pre-processed data
    for participant in participants:
        # Load pre-processed data
        preprocessed_data = mne.io.read_raw_fif('{}-preprocessed-raw.fif'.format(participant), preload=True)
        
        # Save pre-processed data to HDF5 file under a common group
        group = hf.create_group('participant_{}'.format(participant))
        group.create_dataset('data', data=preprocessed_data.get_data())
        group.attrs['participant_id'] = participant
"""
