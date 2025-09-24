from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = load_dataset("ylecun/mnist")

# Access train split directly
train_data = dataset["train"]

# Extract train images and labels as numpy arrays
x_train = (
    np.array([np.array(img) for img in train_data["image"]])
    .reshape(-1, 28 * 28)
    .astype(np.float32)
    / 255.0
)
y_train = np.array(train_data["label"])

# Access test split directly
test_data = dataset["test"]

# Extract test images and labels as numpy arrays
x_test = (
    np.array([np.array(img) for img in test_data["image"]])
    .reshape(-1, 28 * 28)
    .astype(np.float32)
    / 255.0
)
y_test = np.array(test_data["label"])

# Print shapes
print(f"Train data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")
