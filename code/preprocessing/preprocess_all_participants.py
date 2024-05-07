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

folder = '../../../Thesis/openmiir/raw_data'


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

    ica = mne.preprocessing.read_ica('{}-100p_64c-ica.fif'.format(participant),verbose=False)
    print("***********",ica.n_components_)
    #ica.n_components_ = 0.99
    raw_clean = ica.apply(raw)
    raw_clean.save('LONGER-EPOCHS-{}-preprocessed-precICA-raw.fif'.format(participant), overwrite=True)

