import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Read the input image
image = cv2.imread('C:\YOLOv8_and_Canny_Edge\Plate-Character-Recognition-with-Canny-Edge\\license_dataset\\test\\images\\20-cdmx2017policia-c_jpg.rf.6d6a22071b4ca8e7ba471c045c5f3ed8.jpg', cv2.IMREAD_GRAYSCALE)

# Step 3: Apply Gaussian blur to reduce noise (optional but recommended)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 4: Use the Canny edge detection function
threshold1 = 30
threshold2 = 100
edges = cv2.Canny(blurred, threshold1, threshold2)

# Step 5: Display the output image
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
