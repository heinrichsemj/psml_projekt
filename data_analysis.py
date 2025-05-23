import pandas as pd
import numpy as np
import kagglehub
import os
from PIL import Image

# Takes a numpy array of shape (28, 28) and preprocesses it
def preprocess_image(image_array):

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)

    # Rotate 180 degrees counterclockwise
    image = image.rotate(270, expand=True)

    # Mirror the image
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert back to a numpy array
    return np.array(image)

# Takes a list of labels and the number of classes
# Returns a one-hot encoded numpy array
# Example: one_hot_encode(3, 5) -> [0, 0, 1, 0, 0]
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot

def load_emnist_data():
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
            print(files)
            train_file = next((f for f in files if 'emnist-letters-train.csv' in f and f.endswith('.csv')), None)
            test_file = next((f for f in files if 'emnist-letters-test.csv' in f and f.endswith('.csv')), None)
            path_train = os.path.join(path, train_file)
            path_test = os.path.join(path, test_file)
            
        # Load the training dataset (no header)
        df_train = pd.read_csv(path_train, header=None)
        labels_train = df_train.iloc[:, 0].values   # corresponds to a letter
        images_train = df_train.iloc[:, 1:].values  # pixel values of the images

        # Adjust labels to start from 0
        labels_train -= 1
        # print unique labels
        print("Unique labels in training set:", np.unique(labels_train))

        # Normalize pixel values (0-255 -> 0-1)
        images_train = images_train.astype('float32') / 255.0
        images_train = images_train.reshape(-1, 28 * 28)  # -> vectors

        # Load the testing dataset (no header)
        df_test = pd.read_csv(path_test, header=None)
        labels_test = df_test.iloc[:, 0].values # corresponds to a letter
        images_test = df_test.iloc[:, 1:].values  # pixel values of the images

        # Adjust labels to start from 0
        labels_test -= 1
        print("Unique labels in testing set:", np.unique(labels_test))

        # Normalize pixel values (0-255 -> 0-1)
        images_test = images_test.astype('float32') / 255.0
        images_test = images_test.reshape(-1, 28 * 28)  # -> vectors

        # Preprocess images so that they are in the same orientation
        images_train = np.array([preprocess_image(img.reshape(28, 28)).flatten() for img in images_train])
        images_test = np.array([preprocess_image(img.reshape(28, 28)).flatten() for img in images_test])
        labels_train = one_hot_encode(labels_train, len(np.unique(labels_train)))
        labels_test = one_hot_encode(labels_test, len(np.unique(labels_test)))

        # Save preprocessed data
        np.savez("emnist_preprocessed.npz", 
                images_train=images_train, 
                labels_train=labels_train, 
                images_test=images_test, 
                labels_test=labels_test)
        print("Preprocessed data saved to 'emnist_preprocessed.npz'")
    return images_train, labels_train, images_test, labels_test


# Example usage
if __name__ == "__main__":
    images_train, labels_train, images_test, labels_test = load_emnist_data()
    print("Sample training label:", labels_train[0])
    print("Sample testing label:", labels_test[0])
    print("Training data range:", images_train.min(), images_train.max())
    print("Testing data range:", images_test.min(), images_test.max())
    print("Training data shape:", images_train.shape)
    print("Testing data shape:", images_test.shape)