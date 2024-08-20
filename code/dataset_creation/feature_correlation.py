


with h5py.File("MODE_FEATURES.h5", "r") as f:
    features = f['features'][:]
    labels = f['labels'][:]

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

num_windows = 20 
num_trials = features.shape[0] // num_windows
num_features = 10 * 64

assert features.shape[1] == num_features, "Unexpected feature shape"
assert features.shape[0] == num_windows * num_trials, "Unexpected number of samples"

# Compute the correlation matrix
correlation_matrix = np.corrcoef(features, rowvar=False)

# Print the shape of the correlation matrix
print("Correlation matrix shape:", correlation_matrix.shape)
print(correlation_matrix)
np.save("mode_features_CM.npy", correlation_matrix)




### aggregate data over all channels into 1 mean channel
reshaped_features = features.reshape(-1, 64, 10)

# Aggregate the features across channels (e.g., using mean)
aggregated_features = np.mean(reshaped_features, axis=1)

print("Aggregated Features shape:", aggregated_features.shape)  # Should be (41860, 10)

# Compute the correlation matrix on the aggregated features
correlation_matrix = np.corrcoef(aggregated_features, rowvar=False)

# Print the shape of the correlation matrix
print("Correlation matrix shape:", correlation_matrix.shape)  # Should be (10, 10)

# Save the correlation matrix
np.save("mode_features_aggregated_CM.npy", correlation_matrix)
