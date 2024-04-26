import matplotlib.pyplot as plt
import pandas as pd
import random

# Define the number of participants, trials per participant, stimuli, and conditions
participants = ['P01', 'P04', 'P05', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
trials_per_participant = 60
num_trials_per_stimulus = 5
stimuli = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
conditions = ['Perception', 'Imagination_1', 'Imagination_2', 'Imagination_3']

# Create a list to store the data
data = []

# Generate the data for each participant, stimuli, and condition
# Create a list to store the data
data = []

# Generate the data for each participant, stimulus, and condition
for participant in participants:
    for stimulus in stimuli:
        # Shuffle the conditions for each stimulus
        random_conditions = random.sample(conditions, len(conditions))
        for condition_idx, condition in enumerate(random_conditions):
            data.append([participant, stimulus, condition, num_trials_per_stimulus])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Participant', 'Stimulus_ID', 'Condition', 'Num_Trials'])

# Display the DataFrame
print(df)

# Group the DataFrame by Stimulus_ID and Condition and calculate the total number of trials
grouped_df = df.groupby(['Stimulus_ID', 'Condition']).sum().reset_index()

# Create a bar plot
plt.figure(figsize=(12, 6))
for idx, stimulus_id in enumerate(stimuli):
    plt.subplot(3, 4, idx+1)
    subset_df = grouped_df[grouped_df['Stimulus_ID'] == stimulus_id]
    conditions = subset_df['Condition']
    num_trials = subset_df['Num_Trials']
    plt.bar(conditions, num_trials, color='skyblue')
    plt.title(f'Stimulus {stimulus_id}')
    plt.xlabel('Condition')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()