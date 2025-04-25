import numpy as np
from PIL import Image
import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # random weights initialization
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(2. / self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(2. / self.hidden_nodes)
        
        # Activation function for hidden layers (ReLU)
        self.activation_function = lambda x: np.maximum(0, x)

        # Derivative of ReLU for backpropagation
        self.activation_function_derivative = lambda x: np.where(x > 0, 1, 0)

        # Activation function for output layer (softmax)
        self.output_activation_function = lambda x: np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True)
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Forward pass
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.output_activation_function(final_inputs)

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
            (hidden_errors * self.activation_function_derivative(hidden_inputs)),
            inputs.T
        )

        # Gradient clipping
        gradient_clip_value = 1.0
        self.weights_ho = np.clip(self.weights_ho, -gradient_clip_value, gradient_clip_value)
        self.weights_ih = np.clip(self.weights_ih, -gradient_clip_value, gradient_clip_value)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.output_activation_function(final_inputs)

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
    
    def generate_image(self, target_letter, iterations=1000, learning_rate=0.1):
        """
        Generate an image for a given target letter.

        Parameters:
        target_letter (str): The target letter to generate an image for.
        iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for optimization.

        Returns:
        array: The generated image as a 1D array.
        """
        # Convert the target letter to its corresponding one-hot encoded vector
        target_index = target_letter
        target_output = np.zeros(self.output_nodes)
        target_output[target_index] = 1.0

        # Start with a random input image
        generated_image = np.random.rand(self.input_nodes)

        for _ in range(iterations):
            # Forward pass
            inputs = np.array(generated_image, ndmin=2).T
            hidden_inputs = np.dot(self.weights_ih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(self.weights_ho, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            # Calculate the error
            output_errors = target_output - final_outputs.T

            # Backpropagate the error to adjust the input image
            hidden_errors = np.dot(self.weights_ho.T, output_errors.T)
            input_gradients = np.dot(self.weights_ih.T, hidden_errors * hidden_outputs * (1.0 - hidden_outputs))

            # Update the generated image
            generated_image += learning_rate * input_gradients.T.flatten()

            # Clip the pixel values to the range [0, 1]
            generated_image = np.clip(generated_image, 0, 1)
        return generated_image
# Example usage
if __name__ == "__main__":

    # Initialize the neural network
    input_nodes = 784  # 28x28 pixels
    hidden_nodes = 256
    output_nodes = 52  # Assuming 52 classes (A-Z, a-z)
    learning_rate = 0.001

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Load preprocessed data
    data = np.load("emnist_preprocessed.npz")
    images_train = data["images_train"]
    labels_train = data["labels_train"]
    images_test = data["images_test"]
    labels_test = data["labels_test"]

    # Train the network
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(len(images_train)):
            output = nn.query(images_train[i])
            loss = -np.sum(labels_train[i] * np.log(output + 1e-9))  # Cross-entropy loss
            total_loss += loss
            nn.train(images_train[i], labels_train[i])
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Test the network
    correct = 0
    for i in range(len(images_test)):
        output = nn.query(images_test[i])
        predicted_label = np.argmax(output)
        true_label = np.argmax(labels_test[i])
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / len(images_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate an image for the letter 'A' (assuming 'A' corresponds to index 0)
    generated_image = nn.generate_image(target_letter=0, iterations=1000, learning_rate=0.1)

    # Reshape and save the generated image
    generated_image_reshaped = (generated_image * 255).reshape(28, 28).astype(np.uint8)
    image = Image.fromarray(generated_image_reshaped, mode='L')
    image.save("generated_A.png")
    print("Generated image saved as 'generated_A.png'")