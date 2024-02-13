import cv2
import numpy as np


def draw_min_area_rect(mask_path):
    mask_colored = cv2.imread(mask_path)
    gray_image = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2GRAY)
    threshold = 100

    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the minimum area rectangle for the largest contour
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # Get the four points of the box
        box = np.int0(box)  # Convert to integer

        # Draw the rectangle
        cv2.drawContours(mask_colored, [box], 0, (0, 255, 0), 2)  # Green rectangle

        # Show the image
        cv2.imshow('Min Area Rectangle', mask_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError("No contours found in the mask")


# Example usage
mask_path = 'test2.png'
draw_min_area_rect(mask_path)



