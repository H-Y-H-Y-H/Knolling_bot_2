import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def adjust_box(box, center_image, factor):
    center_box = np.mean(box, axis=0)
    move_vector = center_image - center_box
    move_vector *= factor
    new_box = box + move_vector
    return new_box.astype(int)

def find_rotated_bounding_box(mask_channel, center_image, factor=0.05):
    contours, _ = cv2.findContours(mask_channel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        adjusted_box = adjust_box(box, center_image, factor)
        return adjusted_box, rect  
    return None, None

def process_file(filepath, output_dir):
    mask = np.load(filepath)
    center_image = np.array([mask.shape[2] // 2, mask.shape[1] // 2])
    filename = os.path.basename(filepath)
    base_filename = os.path.splitext(filename)[0]
    
    bounding_boxes_info = []
    for i in range(mask.shape[0]):
        channel = mask[i]
        box, rect = find_rotated_bounding_box(channel, center_image)
        if box is not None:
            # Extract rect details for bounding box information
            (x, y), (width, height), angle = rect
            bounding_boxes_info.append([x, y, width, height, angle])
    
    # Save bounding box info to a text file
    output_filepath = os.path.join(output_dir, f"{base_filename}.txt")
    with open(output_filepath, 'w') as f:
        for bbox in bounding_boxes_info:
            f.write(f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}\n")

def process_directory(directory, output_dir):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                filepath = os.path.join(root, file)
                process_file(filepath, output_dir)

directory = 'masks' 
output_dir = 'bounding_boxes'  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
process_directory(directory, output_dir)

