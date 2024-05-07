# functions to import
import mne
import numpy as np
import os
#import xlrd
import pandas as pd

def merge_trial_and_audio_onsets(raw, use_audio_onsets=True, inplace=True, stim_channel='STI 014', verbose=False):
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
    merged = list() # prepare end result
    #last_trial_event = None
    for i, event in enumerate(events):
        etype = event[2]  #event type is in third column of events
        if etype < 1000 or etype == 1111: # trial or noise onset
            if use_audio_onsets and events[i+1][2] == 1000: # followed by audio onset
                onset = events[i+1][0]
                merged.append([onset, 0, etype]) #replace the trial/noise onset event with the audio onset event
                if verbose:
                    print('merged {} + {} = {}'.format(event, events[i+1], merged[-1]))
            else:
                # either we are not interested in audio onsets or there is none
                merged.append(event)
                if verbose:
                    print('kept {}'.format(merged[-1]))
        # audio onsets (etype == 1000) are not copied
        if etype > 1111: # other events (keystrokes)
            merged.append(event)
            if verbose:
                print('kept other {}'.format(merged[-1]))

    merged = np.asarray(merged, dtype=int)

    if inplace:
        stim_id = raw.ch_names.index(stim_channel)
        raw._data[stim_id,:].fill(0)     # delete data in stim channel
        raw.add_events(merged)

    return merged

def load_beat_times(stimulus_id, cue=False, data_root=None, verbose=True, version=1):

    if cue:
        beats_filepath = os.path.join('../../../Thesis/openmiir/meta',
                                      'beats.v{}'.format(version),
                                      '{}_cue_beats.txt'.format(stimulus_id))
    else:
        beats_filepath = os.path.join('../../../Thesis/openmiir/meta',
                                      'beats.v{}'.format(version),
                                      '{}_beats.txt'.format(stimulus_id))

    with open(beats_filepath, 'r') as f:
        lines = f.readlines()

    beats = []
    for line in lines:
        if not line.strip().startswith('#'):
            beats.append(float(line.strip()))
    beats = np.asarray(beats)

    if verbose:
        print('Read {} beat times from {}'.format(len(beats), beats_filepath))

    return beats


def load_stimuli_metadata_map(key=None, data_root=None, verbose=None, version=None):
    STIMULUS_IDS = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
    current = os.getcwd()
    data_root = os.path.join(os.path.dirname(current), 'openmiir/')

    if version is None:
        version = 1

    # handle special case for beats
    if key == 'cue_beats':
        key = 'beats'
        cue = True
    else:
        cue = False

    if key == 'beats':
        map = dict()
        for stimulus_id in STIMULUS_IDS:
            map[stimulus_id] = load_beat_times(stimulus_id,
                                               cue=cue,
                                               data_root=data_root,
                                               verbose=None,
                                               version=version)
        return map

    current = os.getcwd()
    data_root = os.path.join(os.path.dirname(current), '../../../Thesis/openmiir/')
    meta = load_stimuli_metadata(data_root, version=version)

    if key is None:
        return meta  # return everything

    map = dict()
    for stimulus_id in STIMULUS_IDS:
        map[stimulus_id] = meta[stimulus_id][key]

    return map

def default_beat_event_id_generator(stimulus_id, condition, cue, beat_count):
    if cue:
        cue = 0
    else:
        cue = 10
    return 100000 + stimulus_id * 1000 + condition * 100 + cue + beat_count

def load_stimuli_metadata(data_root=None, version=None, verbose=None):

    """if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')

    xlsx_filepath = os.path.join(data_root, 'meta', 'Stimuli_Meta.v{}.xlsx'.format(version))
    """

    xlsx_filepath = '../../../Thesis/openmiir/meta/Stimuli_Meta.v2.xlsx'
    #book = xlrd.open_workbook(xlsx_filepath, encoding_override="cp1252")
    #sheet = book.sheet_by_index(0)
    sheet = pd.read_excel(xlsx_filepath)#, encoding='cp1252') #book=
    #sheet = book[0]

    if verbose:
        print('Loading stimulus metadata from {}'.format(xlsx_filepath))

    meta = dict()
    for i in range(0, 12): 
        """print("0: ", sheet.iloc[i, 0])
        print("1: ", sheet.iloc[i, 1])
        print("2: ", sheet.iloc[i, 2])
        print("3: ", sheet.iloc[i, 3])
        print("4: ", sheet.iloc[i, 4])
        print("5: ", sheet.iloc[i, 5])
        print("6: ", sheet.iloc[i, 6])
        print("7: ", sheet.iloc[i, 7])
        print("8: ", sheet.iloc[i, 8])
        print("11: ", sheet.iloc[i, 11])
        print("14: ", sheet.iloc[i, 14])
        print("15: ", sheet.iloc[i, 15])
        print("16: ", sheet.iloc[i, 16])"""


        #stimulus_id = int(sheet.cell(i,0).value)
        stimulus_id = int(sheet.iloc[i, 0])
        meta[stimulus_id] = {
            'id' : stimulus_id,
            'label' : sheet.iloc[i, 1].encode('ascii'),
            'audio_file' : sheet.iloc[i,2].encode('ascii'),
            #'cue_file' : sheet.cell(i,2).value.encode('ascii').replace('.wav', '_cue.wav'),
            'cue_file' : sheet.iloc[i,2].replace('.wav', '_cue.wav'),
            'length_with_cue' : sheet.iloc[i,3],
            'length_of_cue' : sheet.iloc[i,4],
            'length_without_cue' : sheet.iloc[i,5],
            'length_of_cue_only' : sheet.iloc[i,6],
            'cue_bpm' : int(sheet.iloc[i,7]),
            'beats_per_bar' : int(sheet.iloc[i,8]),
            'num_bars' : int(sheet.iloc[i,14]),
            'cue_bars' : int(sheet.iloc[i,15]),
            'bpm' : int(sheet.iloc[i,16]),
            'approx_bar_length' : sheet.iloc[i,11],
        }

        if version == 2:
            meta[stimulus_id]['bpm'] = meta[stimulus_id]['cue_bpm'] # use cue bpm

    return meta


def decode_event_id(event_id):
    if event_id < 1000:
        #stimulus_id = event_id / 10
        stimulus_id = int(str(event_id)[:-1])
        condition = event_id % 10
        return stimulus_id, condition
    else:
        return event_id
    

def generate_beat_events(trial_events,                  # base events as stored in raw fif files
                         include_cue_beats=True,        # generate events for cue beats as well?
                         use_audio_onset=True,          # use the more precise audio onset marker (code 1000) if present
                         exclude_stimulus_ids=[],
                         exclude_condition_ids=[],
                         beat_event_id_generator=default_beat_event_id_generator,
                         sr=512.0,                      # sample rate, correct value important to compute event frames
                         verbose=False,
                         version=None):
    STIMULUS_IDS = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
    ## prepare return value
    beat_events = []

    ## get stimuli meta information
    meta = load_stimuli_metadata_map() ###
    beats = load_stimuli_metadata_map('beats', verbose=verbose, version=version)
    
    if include_cue_beats: #set to True now, maybe experiment with False
        cue_beats = load_stimuli_metadata_map('cue_beats')

        ## determine the number of cue beats
        num_cue_beats = dict()
        for stimulus_id in STIMULUS_IDS:
            num_cue_beats[stimulus_id] = \
                meta[stimulus_id]['beats_per_bar'] * meta[stimulus_id]['cue_bars']
        if verbose:
            print(num_cue_beats)


    ## helper function to add a single beat event
    def add_beat_event(etime, stimulus_id, condition, beat_count, cue=False):
        etype = beat_event_id_generator(stimulus_id, condition, cue, beat_count)
        beat_events.append([etime, 0, etype])


        if verbose:
            print(beat_events[-1])

    ## helper function to add a batch of beat events
    def add_beat_events(etimes, stimulus_id, condition, cue=False):
        beats_per_bar = meta[stimulus_id]['beats_per_bar']
        for i, etime in enumerate(etimes):
            beat_count = (i % beats_per_bar) + 1
            add_beat_event(etime, stimulus_id, condition, beat_count, cue)

    for i, event in enumerate(trial_events):
        etype = event[2]


        etime = event[0]

        if verbose:
            print('{:4d} at {:8d}'.format(etype, etime))

        if etype >= 1000: # stimulus_id + condition
            continue

        stimulus_id, condition = decode_event_id(etype)
        if stimulus_id in exclude_stimulus_ids or condition in exclude_condition_ids:
            continue  # skip excluded

        trial_start = etime # default: use trial onset
        if use_audio_onset and condition < 3:
            # Note: conditions 3 and 4 have no audio cues
            next_event = trial_events[i+1]
            if next_event[2] == 1000: # only use if audio onset
                trial_start = next_event[0]

        if verbose:
            print('Trial start at {}'.format(trial_start))

        if condition < 3: # cued
            offset = sr * meta[stimulus_id]['length_of_cue']

            if include_cue_beats:
                cue_beat_times = trial_start + np.floor(sr * cue_beats[stimulus_id])
                cue_beat_times = cue_beat_times[:num_cue_beats[stimulus_id]]  # truncate at num_cue_beats
                cue_beat_times = np.asarray(cue_beat_times, dtype=int)
                if verbose:
                    print(cue_beat_times)
                add_beat_events(cue_beat_times, stimulus_id, condition, cue=True)
        else:
            offset = 0 # no cue

        beat_times = trial_start + offset + np.floor(sr * beats[stimulus_id])
        beat_times = np.asarray(beat_times, dtype=int)
        if verbose:
            print(beat_times[:5], '...')
        add_beat_events(beat_times, stimulus_id, condition)

    beat_events = np.asarray(beat_events, dtype=int)

    return beat_events