import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import (
    label,
    regionprops,
    moments,
    moments_central,
    moments_normalized,
    moments_hu,
)
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle


def add_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    th = 190
    img_binary = (image < th).astype(np.double)
    kernel_size = 6
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    modified_image = cv2.dilate(img_binary, kernel, iterations=1)
    binary_image = cv2.erode(modified_image, kernel, iterations=1)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate through each contours
    for i, contour in enumerate(contours):
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a thick red bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Add a larger letter 'A' next to each bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.5
        cv2.putText(
            image, "A", (x + w + 15, y + h // 2), font, font_size, (0, 255, 0), 3
        )

    # Display the result
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace 'image.jpeg' with the path to your image
add_bounding_boxes("a.bmp")
