import os
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab, Image
import numpy as np
from neural_network import NeuralNetwork
import data_analysis

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Für Windows High-DPI
except:
    pass

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
    testbild_def.save('testbilddef.png')

# function for correct loss calculation   
def cross_entropy_loss(targets, outputs):
    epsilon = 1e-9  # Small value to avoid log(0)
    outputs = np.clip(outputs, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(outputs))

# Initialize the neural network
input_nodes = 784  # 28x28 pixels
hidden_nodes = 200
output_nodes = 26   # EMNIST has 52 classes (a-z)
learning_rate = 0.1

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
    try:
        # Accurate canvas capture (with small buffer for borders)
        x = canvas.winfo_rootx() + 2
        y = canvas.winfo_rooty() + 2
        x1 = x + canvas.winfo_width() - 4
        y1 = y + canvas.winfo_height() - 4

        img = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Resize and convert to grayscale
        img = img.resize((28, 28), Image.Resampling.LANCZOS).convert('L')

        """# Invert: black background becomes white (0 → 255), white drawing becomes black
        img = Image.eval(img, lambda x: 255 - x)"""

        # Normalize: 0 (black stroke) → 1.0, 255 (white bg) → 0.0
        im_values = np.asarray(img) / 255.0

        # Flatten to 1D input
        input_data = im_values.flatten()

        output = nn.query(input_data)
        predicted_class = np.argmax(output)
        confidence = float(output[predicted_class][0]) * 100

        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

        prediction_label.config(text=f"Prediction: {alphabet[predicted_class]}")
        confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        status_bar.config(text="Prediction complete")

        visualize(input_data)

    except Exception as e:
        print("Prediction error:", e)
        status_bar.config(text=f"Error: {e}")

# Enhanced clear function
def clear_canvas():
    canvas.delete('all')
    prediction_label.config(text="Draw a digit")
    confidence_label.config(text="Confidence: -")
    status_bar.config(text="Canvas cleared")

# Create a modern, polished tkinter window
window = tk.Tk()
window.title("Digit Recognition App")
# Cross-platform fullscreen
window.geometry(f"{window.winfo_screenwidth()}x{window.winfo_screenheight()}")
try:
    window.state('zoomed')  # Works on Windows
except:
    pass


window.configure(bg='#f0f0f0')

# Set window icon (replace with your own icon if available)
try:
    window.iconbitmap('digit_icon.ico')  # Create or add your own .ico file
except:
    pass  # Skip if icon not found

# Style configuration
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=6)
style.configure('TLabel', font=('Helvetica', 14), background='#f0f0f0')
style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'))

# Create main frames
header_frame = ttk.Frame(window, padding="10")
header_frame.pack(fill=tk.X)

main_frame = ttk.Frame(window)
main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

canvas_frame = ttk.Frame(main_frame)
canvas_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

control_frame = ttk.Frame(main_frame, width=250)
control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

# Header
title_label = ttk.Label(header_frame, text="Handwritten Digit Recognition", style='Title.TLabel')
title_label.pack()

# Drawing canvas
canvas = tk.Canvas(canvas_frame, width=540, height=540, bg="black", highlightthickness=1, 
                  highlightbackground="#cccccc", cursor="pencil")
canvas.pack(expand=True, pady=10)

# Control panel
control_label = ttk.Label(control_frame, text="Controls", style='Title.TLabel')
control_label.pack(pady=(0, 20))

# Brush size control
brush_frame = ttk.LabelFrame(control_frame, text="Brush Settings", padding=10)
brush_frame.pack(fill=tk.X, pady=5)

brush_size = tk.IntVar(value=15)
brush_label = ttk.Label(brush_frame, text="Brush Size:")
brush_label.pack(anchor=tk.W)

brush_slider = ttk.Scale(brush_frame, from_=5, to=30, variable=brush_size, 
                        orient=tk.HORIZONTAL)
brush_slider.pack(fill=tk.X)

# Prediction display
prediction_frame = ttk.LabelFrame(control_frame, text="Prediction", padding=15)
prediction_frame.pack(fill=tk.X, pady=15)

prediction_label = ttk.Label(prediction_frame, text="Draw a digit", 
                            font=('Helvetica', 24), foreground='#2c3e50')
prediction_label.pack()

confidence_label = ttk.Label(prediction_frame, text="Confidence: -", 
                            font=('Helvetica', 12))
confidence_label.pack(pady=(10, 0))

# Buttons
button_frame = ttk.Frame(control_frame)
button_frame.pack(fill=tk.X, pady=10)

recognize_button = ttk.Button(button_frame, text="Recognize", command=predict)
recognize_button.pack(fill=tk.X, pady=5)

clear_button = ttk.Button(button_frame, text="Clear Canvas", command=clear_canvas)
clear_button.pack(fill=tk.X, pady=5)

# Status bar
status_bar = ttk.Label(window, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

# Modified paint function with brush size control
def paint(event):
    brush = brush_size.get()
    x1, y1 = (event.x - brush), (event.y - brush)
    x2, y2 = (event.x + brush), (event.y + brush)
    # Use black color explicitly
    canvas.create_oval(x1, y1, x2, y2, outline="black", fill="white", width=0)
    
canvas.bind("<B1-Motion>", paint)

def close_window(event=None):
    print("Fenster wird geschlossen...")
    window.destroy()  # Closes the Tkinter window
window.bind("<q>", close_window)  # Press 'q' to close the window

# Run the application
window.mainloop()