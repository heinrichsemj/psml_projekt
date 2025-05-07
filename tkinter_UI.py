import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

im_values = []
window = tk.Tk()
window.geometry('820x600')

def visualize(input_image):  
    num_pixels = len(input_image)
    num_rows_columns = int(np.sqrt(num_pixels))
    testbild_def = Image.new('RGB', (num_rows_columns, num_rows_columns), (255, 255, 255))
    for i in range(num_rows_columns):
        for j in range(num_rows_columns):
            coordinate_def = (i, j)
            pixel_value = int(input_image[j * num_rows_columns + i] * 255)
            testbild_def.putpixel(coordinate_def, (pixel_value, pixel_value, pixel_value))
    testbild_def.save('testbilddef.png')

def paint(event, brush_size=15):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, outline="white", fill="white", width=0)
    # Also draw on the PIL image
    draw.ellipse([x1, y1, x2, y2], fill="white")

def clear():
    canvas.delete('all')
    global draw, pil_image
    pil_image = Image.new('RGB', (540, 540), 'black')
    draw = ImageDraw.Draw(pil_image)
    im_values.clear()

def safe_img():
    img_pixel = 64
    global im_values
    # Use the PIL image we've been drawing to
    small_img = pil_image.resize((img_pixel, img_pixel), Image.Resampling.LANCZOS)
    
    im_values.clear()
    for i in range(img_pixel):
        for j in range(img_pixel):
            r, g, b = small_img.getpixel((j, i))[:3]
            brightness = ((r + g + b) // 3)/255.0
            im_values.append(brightness)
    visualize(im_values)
    print("im_values:", im_values)

def close_window(event=None):
    print("window closing...")
    window.destroy()  # Closes the Tkinter window

# Create canvas and buttons
canvas = tk.Canvas(width=540, height=540, bg="black", cursor="cross")
label = tk.Label(text="Zeichne eine Zahl!", font=("Calibri", 20))
classify_btn = tk.Button(text="Recognise", command=safe_img)
button_clear = tk.Button(text="Clear", command=clear)

# Positioning
canvas.grid(row=0, column=0, pady=2, padx=2)
label.grid(row=0, column=1, pady=2, padx=2)
classify_btn.grid(row=1, column=1, pady=2, padx=2)
button_clear.grid(row=1, column=0, pady=2)
canvas.bind("<B1-Motion>", paint)

# Create PIL image to mirror canvas drawings
pil_image = Image.new('RGB', (540, 540), 'black')
draw = ImageDraw.Draw(pil_image)

# Bind the close_window function to the 'q' key
window.bind("<q>", close_window)  # Press 'q' to close the window


window.mainloop()
