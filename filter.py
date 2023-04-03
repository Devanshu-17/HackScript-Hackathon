import cv2
import numpy as np

# Load the image
image_path = 'a.png'
image = cv2.imread(image_path)

# Convert to grayscale and binarize the image using Otsu's thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply morphological operations to remove noise and fill in gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours in the image
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours and detect checkboxes
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w/h
    
    # Check if the aspect ratio of the bounding rectangle is close to 1 (square shape)
    if aspect_ratio > 0.8 and aspect_ratio < 1.2:
        # Extract the checkbox region and binarize it using Otsu's thresholding
        checkbox_region = gray[y:y+h, x:x+w]
        _, checkbox_thresh = cv2.threshold(checkbox_region, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Calculate the percentage of white pixels in the checkbox region
        checkbox_area = w * h
        white_area = np.sum(checkbox_thresh == 255)
        white_ratio = white_area / checkbox_area
        
        # Determine if the checkbox is checked or unchecked based on the white ratio
        if white_ratio > 0.5:
            print("Checkbox is checked")
        else:
            print("Checkbox is unchecked")
