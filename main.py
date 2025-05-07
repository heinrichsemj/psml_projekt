import os
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab, Image
import numpy as np
from neural_network import NeuralNetwork, visualize
from data_analysis import load_emnist_data
from neural_network import train_model

# Check if the script is running on Windows and set DPI awareness
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Für Windows High-DPI
except:
    pass

# Initialize the neural network
input_nodes = 784  # 28x28 pixels
hidden_nodes = 200
output_nodes = 26   # EMNIST has 26 classes (A-Z)
learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Load preprocessed data
images_train, labels_train, images_test, labels_test = load_emnist_data()
# Train the neural network
train_model(nn, images_train, labels_train, epochs=5)
    
# Function to make a prediction
def predict():
    try:
        # Canvas capture
        x = canvas.winfo_rootx() + 2
        y = canvas.winfo_rooty() + 2
        x1 = x + canvas.winfo_width() - 4
        y1 = y + canvas.winfo_height() - 4

        img = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Resize and convert to grayscale
        img = img.resize((28, 28), Image.Resampling.LANCZOS).convert('L')

        # Normalize: 255 (white) → 1.0, 0 (black) → 0.0
        im_values = np.asarray(img) / 255.0

        # Flatten to 1D input
        input_data = im_values.flatten()

        # Get prediction
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


# Tkinter window setup
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