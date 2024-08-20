import h5py
import numpy as np
from collections import Counter

import torch
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'num_classes': [12],
    'num_electrodes': [64],
    'chunk_size': [3538],
    'dropout': [0.25, 0.5], #0.1,0.75
    'lr': [0.0001], #, 0.001, 0.01, 0.1
    'F1': [4, 8, 16, 32],
    'F2': [8, 16, 32, 64]
}

# Load the EEG data from the HDF5 file
data_path = "../dataset_creation/TRIMMED_PERCEPTION.h5"
with h5py.File(data_path, 'r') as f:
    X = f['data'][:]
    subjects = f['subjects'][:]
    y = f['labels'][:]
    conditions = f['condition'][:]

print(X.shape)  # (540, 64, 3538)
print(y.shape)  # (540,)
print(Counter(y))  # 45 per each of the 12 classes

# Map song indices to zero-based indices
label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
mapped_labels = [label_map[label] for label in y]
print(mapped_labels)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(mapped_labels, dtype=torch.long)

print(X_tensor.size())  # (540, 64, 3538)
print(y_tensor.size())  # (540,)

# Add a channel dimension to X_tensor
X_tensor = X_tensor.unsqueeze(1)

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

        # Leave out one participant's data as the test set
        start_idx = 0  # 60*n
        end_idx = 60  # 60*n + 60
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
        test_set = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_set, batch_size=64, num_workers=59)
        val_loader = DataLoader(val_set, batch_size=64, num_workers=59)
        test_loader = DataLoader(test_set, batch_size=64, num_workers=59)

        # Define model with current parameters
        model = EEGNet(num_classes=params["num_classes"], num_electrodes=params["num_electrodes"],
                        chunk_size=params["chunk_size"], dropout=params["dropout"],
                        F1=params["F1"], F2=params["F2"])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Track accuracies for each epoch
        train_accuracies = []
        val_accuracies = []

        # Training loop with accuracy tracking
        for epoch in range(200):  # Max epochs set to 50
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
            train_accuracies.append(train_accuracy.item())

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
            val_accuracies.append(val_accuracy.item())

            print(f"Epoch {epoch+1}/{50}, Train Accuracy: {train_accuracy.item()}, Val Accuracy: {val_accuracy.item()}")
            f.write(f"Epoch {epoch+1}/{50}, Train Accuracy: {train_accuracy.item()}, Val Accuracy: {val_accuracy.item()}\n")

        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        accuracies.append(avg_val_accuracy)
        print("MODEL: ", str(model))
        print("PARAMS: ", params)
        print("Average Validation Accuracy:", avg_val_accuracy)
        f.write("Average Validation Accuracy: " + str(avg_val_accuracy) + "\n")
        print("-------------------------")
    
    # Calculate and store best parameters based on validation accuracy
    best_accuracy = max(accuracies)
    best_params = param_combinations[accuracies.index(best_accuracy)]
    print("BEST ACCURACY: ", best_accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("Average Accuracy:", avg_accuracy)
    f.write("Average Accuracy: " + str(avg_accuracy) + "\n\n")

# After tuning, train the final model on the entire training set (excluding test set) and evaluate on the test set
final_model = EEGNet(num_classes=best_params["num_classes"], num_electrodes=best_params["num_electrodes"],
                     chunk_size=best_params["chunk_size"], dropout=best_params["dropout"],
                     F1=best_params["F1"], F2=best_params["F2"])
final_criterion = nn.CrossEntropyLoss()
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

final_trainer = ClassifierTrainer(model=final_model, num_classes=best_params['num_classes'])
final_trainer.fit(train_loader, val_loader, max_epochs=50)

# Evaluate on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        outputs = final_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    final_accuracy = correct / total
    print("Final Test Accuracy:", final_accuracy)
    f.write("Final Test Accuracy: " + str(final_accuracy) + "\n")
