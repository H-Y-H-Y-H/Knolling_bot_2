import cv2
import numpy as np

# Load your images
image1 = cv2.imread('sim_images/%012d_after.png' % 0)
image2 = cv2.imread('sim_images/%012d_after.png' % 1)
image3 = cv2.imread('sim_images/%012d_after.png' % 2)
image4 = cv2.imread('sim_images/%012d_after.png' % 3)
image5 = cv2.imread('sim_images/%012d_after.png' % 4)
image6 = cv2.imread('sim_images/%012d_after.png' % 5)

images = []
for i in range(6):
    image = cv2.imread('sim_images/%012d_after.png' % i)

    images.append(cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC))
# Ensure all images have the same dimensions (resize or crop as needed)

# Create a canvas to display the images
canvas_height = image1.shape[0] * 2
canvas_width = image1.shape[1] * 3  # You can adjust the number of columns as needed
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Paste the images onto the canvas
canvas[:image1.shape[0], :image1.shape[1]] = image1
canvas[:image2.shape[0], image1.shape[1]:image1.shape[1] * 2] = image2
canvas[:image3.shape[0], image1.shape[1] * 2:image1.shape[1] * 3] = image3

# Repeat the process for the next row of images
canvas[image1.shape[0]:image1.shape[0] * 2, :image4.shape[1]] = image4
canvas[image1.shape[0]:image1.shape[0] * 2, image4.shape[1]:image4.shape[1] * 2] = image5
canvas[image1.shape[0]:image1.shape[0] * 2, image4.shape[1] * 2:image4.shape[1] * 3] = image6

# Display the canvas with all images
# cv2.imshow('Multiple Images', canvas)
#
# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_height, image_width, _ = images[0].shape

# Create a canvas to display the images
canvas_height = 2 * image_height
canvas_width = 3 * image_width
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Paste the images onto the canvas
for i, image in enumerate(images):
    row = i // 3
    col = i % 3
    canvas[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = image

# Function to handle mouse clicks
selected_image_index = None

def mouse_click(event, x, y, flags, param):
    global selected_image_index
    if event == cv2.EVENT_LBUTTONDOWN:
        print('mouse clicked!')
        col = x // image_width
        row = y // image_height
        print('this is col', col)
        print('this is row', row)
        selected_image_index = row * 3 + col
        print(selected_image_index)
    return selected_image_index

# Create a window and set the mouse callback function
cv2.namedWindow('Multiple Images')
cv2.setMouseCallback('Multiple Images', mouse_click)
cv2.resizeWindow('Multiple Images', 800, 600)
# Display the canvas with all images
cv2.imshow('Multiple Images', canvas)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or selected_image_index is not None:
        break

# Close the window
cv2.destroyAllWindows()

# Print the selected image index (0-based)
if selected_image_index is not None:
    print(f"Selected image index: {selected_image_index}")
else:
    print("No image selected.")
