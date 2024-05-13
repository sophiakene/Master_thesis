import h5py
import numpy as np
from collections import Counter

from mne.decoding import Scaler, Vectorizer, CSP
from mne.preprocessing import Xdawn
from mne import create_info

#from mne_features.feature_extraction import FeatureExtractor

import torch
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.model_selection import KFold

from sklearn.model_selection import ParameterGrid

param_grid = {
    'num_classes': [12],
    'num_electrodes': [64],
    'chunk_size': [9248], #[3538],
    'dropout': [0.1,0.25,0.5,0.75], 
    'lr': [0.0001,0.001,0.01,0.1]  
}

data = "../../../Thesis/PERCEPTION_DATASET_W_PREC_ICA.h5"
data = "../dataset_creation/ZERO-PADDED-PERCEPTION-ALL-CHANNELS.h5"

with h5py.File(data, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

print(X.shape) #(540, 64, 3538)
print(y.shape) #540
print(Counter(y)) #45 per each of the 12 classes

### have to map song indices to "real" 0-based indices (0-11)
# Create a dictionary to map unique labels to zero-based indices
label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
# Map labels to zero-based indices using the dictionary
mapped_labels = [label_map[label] for label in y]
print(mapped_labels)

#trying to average into 1 mean channel 21/04
#print("before averaging: ", X.shape)
#X = np.mean(X, axis = 1)
#print("after averaging: ", X.shape)

# Convert data to PyTorch tensors
#X_tensor = torch.tensor(X_features, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(mapped_labels, dtype=torch.long)



print(X_tensor.size()) #[(540,512) after feature extraction] or (540,64,3538) without feature extraction
print(y_tensor.size()) #(540)

print("before unsqueezing: ",X_tensor.shape)
X_tensor = X_tensor.unsqueeze(1)
print("after unsqueezing: ",X_tensor.shape)
print("*************************")
print(X_tensor[0].shape) #([1, 64, 9248])
print(y_tensor[0].shape) #torch.Size([])
print(y_tensor[0]) #tensor(5)
print(y_tensor) #tensor([ 5,  2, 10,  6,...
print("*************************")


best_accuracy = 0.0
best_params = None

param_combinations = list(ParameterGrid(param_grid))

# Iterate over all parameter combinations
with open("torch_results_song_pred.txt", "w") as f: 
    c = 0
    for params in param_combinations:
        c += 1
        print("Trying Param Combination nr. ", c)
        print(params)
        accuracies = []
        #for n in range(9):  # Cross-validation
            # Split data into train and test sets
        start_idx = 0 #60*n
        end_idx = 60 #60*n + 60
        X_train_tensor = torch.cat([X_tensor[:start_idx],X_tensor[end_idx:]])
        X_test_tensor = X_tensor[start_idx : end_idx]
        y_train_tensor = torch.cat([y_tensor[:start_idx], y_tensor[end_idx:]])
        y_test_tensor = y_tensor[start_idx : end_idx]
                                
        train_set = TensorDataset(X_train_tensor, y_train_tensor)
        test_set = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_set, batch_size=64, num_workers=59)
        val_loader = DataLoader(test_set, batch_size=64, num_workers=59)
        
        # Define model with current parameters
        model = EEGNet(num_classes=params["num_classes"], num_electrodes=params["num_electrodes"],
                        chunk_size=params["chunk_size"], dropout=params["dropout"])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Train the model
        trainer = ClassifierTrainer(model=model, num_classes=params['num_classes'])
        trainer.fit(train_loader, val_loader, max_epochs=50)

        # Evaluate the model on the test set
        #model.eval()
        
        # Your code for evaluating accuracy here
        with torch.no_grad():
            # Perform evaluation
            # For example, calculate accuracy
            correct = 0
            total = 0
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            accuracies.append(accuracy)
            print("MODEL: ", str(model))
            print("PARAMS: ", params)
            print("Accuracy:", accuracy)
            f.write("Accuracy: " + str(accuracy) + "\n")  # Write accuracy to file
            print("-------------------------")
    best_accuracy = max(accuracies)
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("Average Accuracy:", avg_accuracy)
    f.write("Average Accuracy: " + str(avg_accuracy) + "\n\n")  # Write average accuracy to file

        # Calculate and store accuracy
    

"""# Calculate average accuracy across all cross-validation folds
avg_accuracy = np.mean(accuracies)

# Update best accuracy and best parameters if current model is better
if avg_accuracy > best_accuracy:
    best_accuracy = avg_accuracy
    best_params = params

print("Best accuracy:", best_accuracy)
print("Best parameters:", best_params)"""



"""for n in range(9): #0: 0,60 / 1: 60, 120 / 2: 120, 180 ... / 8: 480, 540 
start_idx = 60*n
end_idx = 60*n + 60
X_train_tensor = torch.cat([X_tensor[:start_idx],X_tensor[end_idx:]])
X_test_tensor = X_tensor[start_idx : end_idx]
y_train_tensor = torch.cat([y_tensor[:start_idx], y_tensor[end_idx:]])
y_test_tensor = y_tensor[start_idx : end_idx]
                                
train_set = TensorDataset(X_train_tensor, y_train_tensor)
test_set = TensorDataset(X_test_tensor, y_test_tensor)

#print("len test_dataset", len(train_set)) #60
#print("len train_dataset", len(test_set)) #480

train_loader = DataLoader(train_set, batch_size=64)
val_loader = DataLoader(test_set, batch_size=64)

model = EEGNet(num_classes=2, num_electrodes=64, chunk_size=3538, dropout=0.5) #num_electrodes = 64
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = ClassifierTrainer(model=model, num_classes=2)
trainer.fit(train_loader, val_loader, max_epochs=100)

model.eval()"""


