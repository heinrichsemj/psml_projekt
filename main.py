import os
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
from neural_network import NeuralNetwork
import data_analysis

# function for correct loss calculation   
def cross_entropy_loss(targets, outputs):
    epsilon = 1e-9  # Small value to avoid log(0)
    outputs = np.clip(outputs, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(outputs))

# Initialize the neural network
input_nodes = 784  # 28x28 pixels
hidden_nodes = 200
output_nodes = 52   # EMNIST has 52 classes (A-Z, a-z)
learning_rate = 0.001

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Check if weights exist
weights_file = "weights.npz"
if os.path.exists(weights_file):
    nn.load_weights(weights_file)
    print("Weights loaded successfully!")
else:
    print("Weights not found. Training the network...")
    
    # Load preprocessed data
    data = np.load("emnist_preprocessed.npz")
    images_train = data["images_train"]
    labels_train = data["labels_train"]
    images_test = data["images_test"]
    labels_test = data["labels_test"]

    # Train the network
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(len(images_train)):
            output = nn.query(images_train[i])
            # loss = -np.sum(labels_train[i] * np.log(output + 1e-9))  # Cross-entropy loss (no function)
            loss = cross_entropy_loss(labels_train[i], output)
            total_loss += loss
            nn.train(images_train[i], labels_train[i])
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the trained weights
    nn.save_weights(weights_file)
    print(f"Weights saved to '{weights_file}'")
    
# Test the network
def calculate_accuracy():
    correct = 0
    total = len(data_analysis.images_test)

    for i in range(total):
        # Get the input image and the true label
        input_image = data_analysis.images_test[i]
        true_label = np.argmax(data_analysis.labels_test[i])  # Extract the index of the correct class

        # Query the neural network
        output = nn.query(input_image)
        predicted_label = np.argmax(output)  # Extract the index of the predicted class

        # Check if the prediction is correct
        if predicted_label == true_label:
            correct += 1

    # Calculate and print accuracy
    accuracy = correct / total * 100
    print(f"Accuracy on testing data: {accuracy:.2f}%")

calculate_accuracy()

# Function to make a prediction
def predict():
    # Capture the canvas content
    box = (
        canvas.winfo_rootx() + 3,
        canvas.winfo_rooty() + 3,
        canvas.winfo_rootx() + canvas.winfo_width() - 3,
        canvas.winfo_rooty() + canvas.winfo_height() - 3
    )
    grab = ImageGrab.grab(bbox=box)
    grab = grab.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28

    # Convert the image to grayscale and normalize pixel values
    im_values = []
    for i in range(28):
        for j in range(28):
            r, g, b = grab.getpixel((j, i))[:3]
            brightness = abs(((r + g + b) // 3)) / 255.0
            im_values.append(brightness)
    # Query the neural network
    output = nn.query(im_values)
    predicted_class = np.argmax(output)
    alphabet = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    # Display the prediction on the tkinter window
    prediction_label.config(text=f"Prediction: {alphabet[predicted_class]}")  # Convert to character

# Function to clear the canvas
def clear_canvas():
    canvas.delete('all')
    prediction_label.config(text="Prediction: ")

# Create the tkinter window
window = tk.Tk()
window.geometry('820x600')
window.title("Handwritten Character Recognition")

# Create the canvas for drawing
canvas = tk.Canvas(window, width=540, height=540, bg="black", cursor="cross")
canvas.grid(row=0, column=0, pady=2, padx=2)

# Add buttons and labels
prediction_label = tk.Label(window, text="Prediction: ", font=("Calibri", 20))
prediction_label.grid(row=0, column=1, pady=2, padx=2)

recognize_button = tk.Button(window, text="Recognise", command=predict)
recognize_button.grid(row=1, column=1, pady=2, padx=2)

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=0, pady=2)

# Function to draw on the canvas
def paint(event, brush_size=15):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, outline="white", fill="white", width=0)

canvas.bind("<B1-Motion>", paint)

# Run the tkinter main loop
window.mainloop()