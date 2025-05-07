import numpy as np
from PIL import Image
import numpy as np
import scipy
import os
# From brightness Pixel Vector to image again (image gets saved on "testbilddef.png")
def visualize(input_image):  
    num_pixels = len(input_image)
    num_rows_columns = int(np.sqrt(num_pixels))
    testbild_def = Image.new('RGB', (num_rows_columns, num_rows_columns), (255, 255, 255))
    for i in range(num_rows_columns):  # x
        for j in range(num_rows_columns):  # y
            coordinate_def = (i, j)
            pixel_value = int(input_image[j * num_rows_columns + i] * 255)  # Scale back to 0-255
            testbild_def.putpixel(coordinate_def, (pixel_value, pixel_value, pixel_value))
    testbild_def.save('testbild.png')

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # random weights initialization
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(2. / self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(2. / self.hidden_nodes)
        
        # Activation function for hidden layers (ReLU or sigmoid)
        # self.activation_function = lambda x: np.maximum(0, x)
        self.activation_function = lambda x: scipy.special.expit(x)

        # Derivative of ReLU for backpropagation
        # self.activation_function_derivative = lambda x: np.where(x > 0, 1, 0)
        self.activation_function_derivative = lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x))
        # Activation function for output layer (softmax)
        # self.output_activation_function = lambda x: np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Forward pass
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # final_outputs = self.output_activation_function(final_inputs)
        final_outputs = self.activation_function(final_inputs)

        # Backward pass
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Update weights between hidden and output layers
        self.weights_ho += self.learning_rate * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            hidden_outputs.T
        )
        
        # Update weights between input and hidden layers
        self.weights_ih += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            inputs.T
        )
        """
        # Gradient clipping
        gradient_clip_value = 1.0
        self.weights_ho = np.clip(self.weights_ho, -gradient_clip_value, gradient_clip_value)
        self.weights_ih = np.clip(self.weights_ih, -gradient_clip_value, gradient_clip_value)
        """

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # final_outputs = self.output_activation_function(final_inputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    
    def set_weights(self, weights_ih, weights_ho):
        """
        Set weights manually
        
        Parameters:
        weights_ih (array): Input-to-hidden weights
        weights_ho (array): Hidden-to-output weights
        """
        self.weights_ih = weights_ih
        self.weights_ho = weights_ho
    
    def save_weights(self, filename):
        """
        Save weights to a file
        
        Parameters:
        filename (str): Path to save weights
        """
        np.savez(filename, weights_ih=self.weights_ih, weights_ho=self.weights_ho)
    
    def load_weights(self, filename):
        """
        Load weights from a file
        
        Parameters:
        filename (str): Path to load weights from
        """
        data = np.load(filename)
        self.weights_ih = data['weights_ih']
        self.weights_ho = data['weights_ho']
    
    

# Example usage
if __name__ == "__main__":

    # Initialize the neural network
    input_nodes = 784  # 28x28 pixels
    hidden_nodes = 200
    output_nodes = 26  # Assuming 52 classes (A-Z, a-z)
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Load preprocessed data
    data = np.load("emnist_preprocessed.npz")
    images_train = data["images_train"]
    labels_train = data["labels_train"]
    images_test = data["images_test"]
    labels_test = data["labels_test"]
    
    # Initialize the neural network
    input_nodes = 784  # 28x28 pixels
    hidden_nodes = 200
    output_nodes = 26  # Assuming 10 classes (0-9)
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


    # Check if weights exist
    weights_file = "weights.npz"
    if os.path.exists(weights_file):
        nn.load_weights(weights_file)
        print("Weights loaded successfully!")
    else:
        # Train the network
        epochs = 6
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in range(len(images_train)):
                nn.train(images_train[i], labels_train[i])
            correct = 0
            for i in range(len(images_test)):
                output = nn.query(images_test[i])
                predicted_label = int(np.argmax(output))
                true_label = int(np.argmax(labels_test[i]))
                if predicted_label == true_label:
                    correct += 1
            accuracy = correct / len(images_test) * 100
            print(f"Test Accuracy: {accuracy:.2f}%")
        # Save the trained weights
        nn.save_weights("weights.npz")
        print("Weights saved to 'weights.npz'")

    # Test the network
    correct = 0
    for i in range(len(images_test)):
        output = nn.query(images_test[i])
        predicted_label = int(np.argmax(output))
        true_label = int(np.argmax(labels_test[i]))
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / len(images_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    