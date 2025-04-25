import tkinter as tk
from PIL import ImageGrab
from PIL import Image
import numpy as np
from neural_network import NeuralNetwork
import data_analysis

# Initialize the neural network
input_nodes = 784  # 28x28 pixels
hidden_nodes = 200
output_nodes = 52   # EMNIST has 52 classes (A-Z, a-z)
learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
nn.load_weights("weights.npz")  # Load pre-trained weights
print("Weights loaded successfully!")

# Generate an image for the letter 'a'
generated_image = nn.generate_image(0, iterations=1000, learning_rate=0.1)
# Reshape the generated image to 28x28 and save it
from PIL import Image
import numpy as np

generated_image_reshaped = (generated_image * 255).reshape(28, 28).astype(np.uint8)
image = Image.fromarray(generated_image_reshaped, mode='L')
image.save("generated_a.png")
print("Generated image saved as 'generated_a.png'")

def calculate_accuracy():
    correct = 0
    total = len(data_analysis.images_test)

    for i in range(total):
        # Get the input image and the true label
        input_image = data_analysis.images_test[i]
        true_label = data_analysis.labels_test[i]

        # Query the neural network
        output = nn.query(input_image)
        predicted_label = np.argmax(output)

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