import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import h5py
import numpy as np
from sklearn import svm
from mne_features.feature_extraction import FeatureExtractor
from collections import Counter
import seaborn as sns
import sys

import warnings
warnings.filterwarnings("ignore")

condition = sys.argv[1]
baseline_or_ceiling = sys.argv[2]

#change paths here
if condition == "Perception":
    data = "../../../Thesis/INDIVIDUAL_LENGTH_PERCEPTION.h5" #actually not completely sure if this is zero-padded 
    data = "../dataset_creation/ZERO-PADDED-PERCEPTION.h5"
    #trimmed
    data = "../dataset_creation/TRIMMED_PERCEPTION.h5"
elif condition == "Imagination":
    data = "../../../Thesis/INDIVIDUAL_LENGTH_IMAGINATION.h5"
    data = "../dataset_creation/TRIMMED_IMAGINATION.h5"
elif condition == "Both":
    data = "../../../Thesis/INDIVIDUAL_LENGTH_BOTH_CONDITIONS.h5"
    #data = "../dataset_creation/ZERO-PADDED-BOTH-CONDITIONS.h5"
    data = "../dataset_creation/TRIMMED_PERCEPTION.h5"

else:
    raise ValueError("Condition argument not recognized. Please provide 'Perception', 'Imagination', or 'Both'.")


with h5py.File(data, 'r') as f:
    # List all datasets in the file
    print("Datasets in h5 file: ", f.keys())
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

print("shape of data: ", X.shape)
print("Aggregating EEG data into 1 mean channel") #commenting out for zero-padded experiment
X = np.mean(X, axis=1) #don't aggregate over channels when doing feature extraction
#X = X.ravel()
print("shape of data after aggregation: ", X.shape)

# Sample EEG features DataFrame (replace this with your actual data)
# Assuming eeg_features contains features extracted from EEG data for each stimulus
#eeg_features = pd.DataFrame(X)  # Your EEG features data #This is raw data now, just averaged into 1 channel...

# Sample ground truth labels for each classification subtask
meter_translation = {1:0,2:0,3:1,4:1,
                11:0,12:0,13:1,14:1,
                21:0,22:0,23:1,24:1}
lyrics_translation = {1:1,2:1,3:1,4:1,
                11:0,12:0,13:0,14:0,
                21:0,22:0,23:0,24:0}
mode_translation = {1: 0, 2: 1, 3: 1, 4: 1, 
                    11: 0, 12: 1, 13: 1, 14: 1,
                    21: 1, 22: 0, 23: 1, 24: 1}


meter_labels = np.array([meter_translation[label] for label in y]) 
lyrics_labels = np.array([lyrics_translation[label] for label in y])
mode_labels = np.array([mode_translation[label] for label in y]) 


#print(meter_labels.shape)
# Combine ground truth labels into a single DataFrame
# Each row corresponds to a stimulus, and columns represent the labels of each subtask
GT_df = pd.DataFrame({
    'Meter': meter_labels,
    'Lyrics': lyrics_labels,
    'Mode': mode_labels,

})
#print(GT_df[:10])
print("number of data points (trials): ",len(GT_df))

#add EEG data to dataframe
X = X.T #transpose X to shape (3538,540)
for i in range(X.shape[0]):
    GT_df[f'time_point{i+1}'] = X[i]

#X = np.concatenate((X, meter_labels, lyrics_labels, mode_labels))
#X is the EEG data 

### FEATURE EXTRACTION (in this case don't use channel average)
"""fe = FeatureExtractor(selected_funcs=['mean', 'pow_freq_bands']) #just two features to test:)
X = fe.fit_transform(X)"""


# Split the data into training and testing sets
# replace with doing leave-one-participant-out cv
"""X_train, X_test, y_train, y_test = train_test_split(GT_df, y, test_size=0.2, random_state=42, shuffle=False) #maybe stratified as the classes aren't that uniformly distributed...
print("y_train: ", Counter(y_train))
cv = GroupKFold(n_splits=9)
print(cv.get_n_splits(X, y, groups))
for i, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")



baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
#scores = cross_val_score(baseline_classifier, X, y, cv=9) #apparently shuffle=False
baseline_classifier.fit(X_train, y_train)
y_preds = baseline_classifier.predict(X_test)
#print(y_preds)
print(accuracy_score(y_preds, y_test))

#print([(p,t) for (p,t) in zip(y_preds, y_test)])
cf_matrix = confusion_matrix(y_test, y_preds)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)"""


##########################
print("Training + Predicting ...")
#leave-one-participant-out cv
df = GT_df
df["label"] = y
#print("df with labels: ", df) #this looks good

#print("here: ",df.iloc[:5,:3]) #only correct labels

if condition == "Perception":
    fold_size = 60
elif condition == "Imagination":
    fold_size = 170
elif condition == "Both":
    fold_size = 230 
num_folds = 9
total_rows = len(df)
rows_per_fold = total_rows // num_folds

# Initialize lists to store training and testing indices
train_indices = []
test_indices = []

# Generate indices for each fold
for i in range(num_folds):
    # Calculate start and end indices for the test set
    start_index = i * rows_per_fold
    end_index = start_index + fold_size 
    # Generate test indices
    test_idx = list(range(start_index, end_index))
    # Generate train indices (remaining indices)
    train_idx = [idx for idx in range(total_rows) if idx not in test_idx]
    
    # Append indices to lists
    train_indices.append(train_idx)
    test_indices.append(test_idx)

# Perform cross-validation
for fold in range(num_folds):
    # Get train and test sets
    train_set = df.iloc[train_indices[fold]]
    test_set = df.iloc[test_indices[fold]]

    if baseline_or_ceiling == "Baseline":
        #exclude correct labels if Baseline
        X_train = train_set.iloc[:, 3:-1]
        y_train = train_set.iloc[:,-1]
        X_test = test_set.iloc[:,3:-1]
        y_test = test_set.iloc[:,-1]
    elif baseline_or_ceiling == "Ceiling":
        X_train = train_set.iloc[:, :-1]
        y_train = train_set.iloc[:,-1]
        X_test = test_set.iloc[:,:-1]
        y_test = test_set.iloc[:,-1]
    print(X_train.shape)

    #print("lengths of sets: ", len(X_train), len(y_train), len(X_test), len(y_test))

    
    # Your cross-validation logic here, train_model(train_set), evaluate_model(test_set), etc.
    print(f"Fold {fold+1}: Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    baseline_classifier = svm.SVC(kernel = "linear", C=0.0001)
    baseline_classifier.fit(X_train, y_train)
    y_preds = baseline_classifier.predict(X_test)
    #print(y_preds)
    print("Accuracy: ", accuracy_score(y_preds, y_test))

    #print([(p,t) for (p,t) in zip(y_preds, y_test)])

    print("confusion matrix: ", confusion_matrix(y_test, y_preds))



