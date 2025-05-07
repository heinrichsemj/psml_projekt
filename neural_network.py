import numpy as np
from PIL import Image
import numpy as np
import scipy
import os
from data_analysis import load_emnist_data


# Neural Network Class with one hidden layer
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # random weights initialization
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(2. / self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(2. / self.hidden_nodes)
        
        # Activation function for hidden layers (sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)

        # Derivative of sigmoid for backpropagation
        self.activation_function_derivative = lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x))
       
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Forward pass
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
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

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        # Forward pass (same as in training)
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    
    def set_weights(self, weights_ih, weights_ho):

        # Set weights directly
        self.weights_ih = weights_ih
        self.weights_ho = weights_ho
    
    def save_weights(self, filename):

        # Save weights to a file
        np.savez(filename, weights_ih=self.weights_ih, weights_ho=self.weights_ho)
    
    def load_weights(self, filename):
        # Load weights from a file
        data = np.load(filename)
        self.weights_ih = data['weights_ih']
        self.weights_ho = data['weights_ho']
    

# From brightness Pixel Vector (1d) to image (image gets saved on "testbild.png")
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

# Function for accuracy calculation
def test_model(nn, images_test, labels_test):
    # Test the network
    correct = 0
    for i in range(len(images_test)):
        output = nn.query(images_test[i])
        predicted_label = int(np.argmax(output))
        true_label = int(np.argmax(labels_test[i]))
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / len(images_test) * 100
    return accuracy

def train_model(nn, images_train, labels_train, epochs):
    # Check if weights exist
    weights_file = "weights.npz"
    if os.path.exists(weights_file):
        nn.load_weights(weights_file)
        print("Weights loaded successfully!")
    else:
        # Train the network
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in range(len(images_train)):
                nn.train(images_train[i], labels_train[i])

        # Save the trained weights
        nn.save_weights("weights.npz")
        print("Weights saved to 'weights.npz'")


# Example usage
if __name__ == "__main__":

    # Initialize the neural network
    input_nodes = 784  # 28x28 pixels
    hidden_nodes = 200
    output_nodes = 26  # Assuming 26 classes (A-Z)
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Load preprocessed data
    images_train, labels_train, images_test, labels_test = load_emnist_data()

    # Train the model
    train_model(nn, images_train, labels_train, epochs=5)

    # Test the model
    accuracy = test_model(nn, images_test, labels_test)
    print(f"Model accuracy: {accuracy:.2f}%")
    
    