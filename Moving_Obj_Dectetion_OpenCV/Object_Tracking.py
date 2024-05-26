import cv2
import numpy as np

def detect_moving_objects(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        return

    # Get the video frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    # Loop through the video frames
    while True:
        ret, prev_frame = cap.read()  
        ret, curr_frame = cap.read()  

        if not ret:
            break  
        # Convert frames to grayscale 
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # difference between frames
        frame_diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply thresholding to get binary image of moving pixels
        _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        erode = cv2.erode(thresh, None, iterations=3)
        dilate = cv2.dilate(erode, None, iterations=30)

        # Find contours of moving objects
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes 
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(curr_frame)


    print(f"Processed video saved at: {output_path}")

# Specify the input and output paths
video_path = r"clip10.mp4"
output_path = r"Object_Tracking_Output\clip10_output.mp4"

# Call the function to detect moving objects and save the output video
detect_moving_objects(video_path, output_path)