import tkinter as tk
from PIL import ImageGrab
from PIL import Image
import numpy as np

im_values = []
# Erstellt einen Fenster für die Zeichnungen
window = tk.Tk()
window.geometry('820x650')

def visualize(input_image):  # From brightness Pixel Vector to image again (image gets saved on "testbilddef.png")
    num_pixels = len(input_image)
    num_rows_columns = int(np.sqrt(num_pixels))
    testbild_def = Image.new('RGB', (num_rows_columns, num_rows_columns), (255, 255, 255))
    for i in range(num_rows_columns):  # column, x
        for j in range(num_rows_columns):  # row, y
            coordinate_def = (i, j)
            pixel_value = int(input_image[j * num_rows_columns + i] * 255)  # Scale back to 0-255
            testbild_def.putpixel(coordinate_def, (pixel_value, pixel_value, pixel_value))
    print("testbilddef.png")
    testbild_def.save('testbilddef.png')

# Funktion fürs Zeichnen der Zahl
def paint(event, brush_size=15):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, outline="white", fill="white", width=0)


# Löschen der gezeichneten Zahl
def clear():
    canvas.delete('all')


def safe_img():
    box = (
        canvas.winfo_rootx() + 3,  # Adjust for potential border/padding
        canvas.winfo_rooty() + 3,  # Adjust for potential border/padding
        canvas.winfo_rootx() + canvas.winfo_width() - 3,  # Adjust for potential border/padding
        canvas.winfo_rooty() + canvas.winfo_height() - 3  # Adjust for potential border/padding
    )
    grab = ImageGrab.grab(bbox=box)
    grab = grab.resize((28, 28), Image.Resampling.LANCZOS) # Resize to 28x28

    for i in range(28):  # Loop over the resized image dimensions
        for j in range(28):
            r, g, b = grab.getpixel((j, i))[:3]  # Get RGB values
            brightness = abs(((r + g + b) // 3))/255.0  # Calculate brightness
            im_values.append(brightness)
    visualize(im_values)

# Erstellt ein Fenster für das Zeichnen der Zahl
canvas = tk.Canvas(width=540, height= 540, bg="black", cursor="cross")
label = tk.Label(text="Zeichne eine Zahl!", font=("Calibri", 20))
classify_btn = tk.Button(text="Recognise", command=safe_img)
button_clear = tk.Button(text="Clear", command=clear)

# Position des Fenster fürs zeichnen.
canvas.grid(row=0, column=0, pady=2, padx=2)
label.grid(row=0, column=1, pady=2, padx=2)
classify_btn.grid(row=1, column=1, pady=2, padx=2)
button_clear.grid(row=1, column=0, pady=2)
canvas.bind("<B1-Motion>", paint)

window.mainloop()