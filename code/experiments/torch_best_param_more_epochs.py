import h5py
import numpy as np
from collections import Counter

import torch
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load the EEG data from the HDF5 file
data_path = "../dataset_creation/TRIMMED_PERCEPTION.h5"
data_path = "../dataset_creation/TRIMMED_ALL_CONDITIONS.h5"
data_path = "../dataset_creation/TRIMMED_IMAGINATION.h5"
with h5py.File(data_path, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

print(X.shape)  # (540, 64, 3538)
print(y.shape)  # (540,)
print(Counter(y))  # 45 per each of the 12 classes
print(Counter(conditions)) #less for condition 4, imagination wo cue with feedback
print("Conditions for P01: ", Counter(conditions[:240])) #these are the conditions for 1 participant (or should be)
#60 per condition, great (I think they answered yes to all feedback questions)
print(len(conditions)) #2093 for both conditions, good


# Map song indices to zero-based indices
label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
mapped_labels = [label_map[label] for label in y]
#print(mapped_labels)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(mapped_labels, dtype=torch.long)
conditions_tensor = torch.tensor(conditions, dtype=torch.long) #for printing out at the end

print(X_tensor.size())  # (540, 64, 3538)
print(y_tensor.size())  # (540,)

# Add a channel dimension to X_tensor
X_tensor = X_tensor.unsqueeze(1)

# Leave out one participant's data as the test set
start_idx = 0  # 60*n
end_idx = 180 #240 for both conditions #60 for perception!! #180 for imagination  # 60*n + 60
X_train_tensor = torch.cat([X_tensor[:start_idx], X_tensor[end_idx:]])
y_train_tensor = torch.cat([y_tensor[:start_idx], y_tensor[end_idx:]])
X_test_tensor = X_tensor[start_idx:end_idx]
y_test_tensor = y_tensor[start_idx:end_idx]



# Split the training data into training and validation sets for hyperparameter tuning
val_split = int(0.8 * len(X_train_tensor))  # 80-20 split for training and validation
X_val_tensor = X_train_tensor[val_split:]
y_val_tensor = y_train_tensor[val_split:]
X_train_tensor = X_train_tensor[:val_split]
y_train_tensor = y_train_tensor[:val_split]

train_set = TensorDataset(X_train_tensor, y_train_tensor)
val_set = TensorDataset(X_val_tensor, y_val_tensor)
#print("here: ", conditions[start_idx:end_idx]) #perfect, 1,2,3....,4,4,4
test_set = TensorDataset(X_test_tensor, y_test_tensor, conditions_tensor[start_idx:end_idx])

train_loader = DataLoader(train_set, batch_size=64, num_workers=59)
val_loader = DataLoader(val_set, batch_size=64, num_workers=59)
test_loader = DataLoader(test_set, batch_size=64, num_workers=59)

# Define the model with the best parameters
best_params = {
    "num_classes": 12,
    "num_electrodes": 64,
    "chunk_size": 3538,
    "dropout": 0.25,
    "lr": 0.0001,
    "F1": 32,
    "F2": 16
}

# Define model with best parameters
model = EEGNet(num_classes=best_params["num_classes"], num_electrodes=best_params["num_electrodes"],
               chunk_size=best_params["chunk_size"], dropout=best_params["dropout"],
               F1 = best_params["F1"], F2 = best_params["F2"])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# Training loop with accuracy tracking
for epoch in range(200):  # Max epochs set to 200
    model.train()
    running_corrects = 0
    running_total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        running_total += labels.size(0)

    train_accuracy = running_corrects.double() / running_total
    print(f"Epoch {epoch+1}/{200}, Train Accuracy: {train_accuracy.item()}")

model.eval()
val_corrects = 0
val_total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        val_corrects += torch.sum(preds == labels.data)
        val_total += labels.size(0)

val_accuracy = val_corrects.double() / val_total
print(f"Validation Accuracy: {val_accuracy.item()}")

# Evaluate on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels, conds = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Create a dictionary to map label indices back to original labels if needed
        idx_to_label = {idx: label for label, idx in label_map.items()}
        for i in range(len(labels)):
                print(f"Prediction: {idx_to_label[predicted[i].item()]}, Actual Label: {idx_to_label[labels[i].item()]}, Condition: {conds[i].item()}")
    
    final_accuracy = correct / total
    print("Final Test Accuracy:", final_accuracy)