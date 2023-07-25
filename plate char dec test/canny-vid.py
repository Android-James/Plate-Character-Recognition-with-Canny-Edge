import cv2
import numpy as np

def canny_edge_detection(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return edges

# Replace 'path_to_recorded_video' with the actual path to your video file
video_path = 'C:\YOLOv8_and_Canny_Edge\Plate-Character-Recognition-with-Canny-Edge\\license_dataset\\videos\\testVideo5.mp4'
output_path = 'C:\YOLOv8_and_Canny_Edge\Plate-Character-Recognition-with-Canny-Edge\\license_dataset\\videos\\cannyTest.mp4'  # Modify this for the output video

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for H.264 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Process each frame of the video
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Apply Canny edge detection to the frame
    edges = canny_edge_detection(frame)

    # Write the processed frame to the output video
    out.write(edges)

    # Display the original video with Canny edges (optional, for real-time visualization)
    cv2.imshow('Canny Edge Detection', edges)

    # Press 'q' to stop processing the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
video_capture.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
