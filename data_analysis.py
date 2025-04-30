import pandas as pd
import numpy as np
import kagglehub
import os

# Check if preprocessed data exists
if os.path.exists("emnist_preprocessed.npz"):
    print("Loading preprocessed data...")
    data = np.load("emnist_preprocessed.npz")
    images_train = data["images_train"]
    labels_train = data["labels_train"]
    images_test = data["images_test"]
    labels_test = data["labels_test"]
    print("Preprocessed data loaded successfully!")
else:
    print("Preprocessed data not found. Loading and processing EMNIST dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("crawford/emnist")
    print("Loading EMNIST dataset...")

    # Find the training CSV file in the downloaded directory
    if os.path.isdir(path):
        files = os.listdir(path)
        train_file = next((f for f in files if 'emnist-letters-train.csv' in f and f.endswith('.csv')), None)
        test_file = next((f for f in files if 'emnist-letters-test.csv' in f and f.endswith('.csv')), None)
        path_train = os.path.join(path, train_file)
        path_test = os.path.join(path, test_file)
        
    # Load the training dataset (no header)
    df_train = pd.read_csv(path_train, header=None)
    labels_train = df_train.iloc[:, 0].values  # initially a # (0 - 25)
    images_train = df_train.iloc[:, 1:].values 

    # Adjust labels to start from 0
    labels_train -= 1

    # Normalize pixel values (0-255 -> 0-1)
    images_train = images_train.astype('float32') / 255.0
    images_train = images_train.reshape(-1, 28 * 28)  # -> vectors

    # Load the testing dataset (no header)
    df_test = pd.read_csv(path_test, header=None)
    labels_test = df_test.iloc[:, 0].values # initially a # (0 - 25)
    images_test = df_test.iloc[:, 1:].values  

    # Adjust labels to start from 0
    labels_test -= 1

    # Normalize pixel values (0-255 -> 0-1)
    images_test = images_test.astype('float32') / 255.0
    images_test = images_test.reshape(-1, 28 * 28)  # -> vectors

    # Save preprocessed data
    np.savez("emnist_preprocessed.npz", 
             images_train=images_train, 
             labels_train=labels_train, 
             images_test=images_test, 
             labels_test=labels_test)
    print("Preprocessed data saved to 'emnist_preprocessed.npz'")

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot
labels_train = one_hot_encode(labels_train, 26)
labels_test = one_hot_encode(labels_test, 26)

# Example usage
if __name__ == "__main__":
    print("Sample training label:", labels_train[0])
    print("Sample testing label:", labels_test[0])
    print("Training data range:", images_train.min(), images_train.max())
    print("Testing data range:", images_test.min(), images_test.max())
    print("Training data shape:", images_train.shape)
    print("Testing data shape:", images_test.shape)