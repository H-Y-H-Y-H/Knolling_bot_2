import cv2
import numpy as np


Below_this_number_with_be_seg = 200 # Gray
def segment_objects_and_visualize(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray Image with Object IDs", gray)


    _, thresh = cv2.threshold(gray, Below_this_number_with_be_seg, 255, cv2.THRESH_BINARY_INV)


    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare a binary image for visualization
    binary_image = np.zeros_like(image)
    objects_coordinates = []

    for i, contour in enumerate(contours):
        # Extract objects
        print(i)
        x, y, w, h = cv2.boundingRect(contour)
        objects_coordinates.append((x, y, w, h))

        # Draw each object on the binary image with a unique number
        cv2.drawContours(binary_image, [contour], -1, (255, 255, 255), -1)
        cv2.putText(binary_image, str(i + 1), (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the original and the binary image
    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Image with Object IDs", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return objects_coordinates


objects_coordinates = segment_objects_and_visualize('0.jpg')
print(objects_coordinates)
