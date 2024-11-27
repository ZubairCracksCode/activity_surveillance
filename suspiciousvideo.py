# import cv2
# import numpy as np
# from ultralytics import YOLO
# import cvzone
# import os


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         point = [x, y]
#         print(point)

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

# # Load the YOLOv8 model
# model = YOLO("yolo11x.pt")
# names = model.model.names

# # Open the video file (use video file or webcam, here using webcam)
# cap = cv2.VideoCapture('D:\activity_surveillance\action_class\note_giving\note-giving-10 (trimmed).mp4')
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# blur_ratio = 50

# # Ensure frame size matches when writing the video
# frame_size = (1020, 600)

# # Use a codec that matches the file format
# video_writer = cv2.VideoWriter("D:\activity_surveillance\Suspecious_Video\note-giving-10.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

# # Variable to store the user-selected track_id
# selected_track_id = None
# blur_all = True  # Start with all objects blurred

# count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     count += 1
#     if count % 15 != 0:
#         continue

#     # Resize frame to the correct dimensions
#     frame = cv2.resize(frame, frame_size)
#     frame1 = frame.copy()

#     # Run YOLOv8 tracking on the frame, persisting tracks between frames
#     results = model.track(frame, persist=True, classes=0)

#     # Check if there are any boxes in the results
#     if results[0].boxes is not None and results[0].boxes.id is not None:
#         # Get the boxes (x1, y1, x2, y2), class IDs, track IDs, and confidences
#         boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
#         class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
#         track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
#         confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

#         for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
#             c = names[class_id]
#             x1, y1, x2, y2 = box

#             # Extract the region of interest (ROI) from the frame
#             roi = frame[y1:y2, x1:x2]

#             # Apply blur if blur_all is enabled or if the track_id is not the selected one
#             if blur_all or (selected_track_id is not None and track_id != selected_track_id):
#                 # Apply blur to the ROI
#                 blur_obj = cv2.blur(roi, (blur_ratio, blur_ratio))
#                 # Place the blurred ROI back into the original frame
#                 frame[y1:y2, x1:x2] = blur_obj

#             # Draw rectangle around the object (whether blurred or not)
#             color = (0, 255, 0) if track_id == selected_track_id else (0, 0, 255)  # Green for unblurred, red for blurred
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cvzone.putTextRect(frame, f'Track ID: {track_id}', (x1, y2 + 20), 1, 1)
#             cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

#     # Display the result
#     cv2.imshow("RGB", frame)
#     cv2.imshow("FRAME", frame1)

#     # Write the processed frame to the video writer
#     video_writer.write(frame)

#     # Check for key presses
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('s'):
#         # Ask user for the track_id to keep unblurred when 's' is pressed
#         try:
#             selected_track_id = int(input("Enter the track_id to keep unblurred: "))
#             blur_all = False  # Disable full blur mode
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")
#             selected_track_id = None  # Reset the selection if input is invalid
#     elif key == ord('n'):
#         # When 'n' is pressed, blur all objects again
#         blur_all = True
#         selected_track_id = None  # Clear the selected track_id when blurring all objects

# # Release the video capture object and close the display window
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import cvzone
# import os

# # Set environment variable to avoid OpenCV duplicate library error
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Callback function for mouse events (optional, can be removed if not needed)
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         point = [x, y]
#         print(point)

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

# # Load the YOLOv8 model
# model = YOLO("yolo11x.pt")
# names = model.model.names  # Get class names

# # Video input/output paths
# input_path = 'D:/activity_surveillance/action_class/note_giving/note-giving-10 (trimmed).mp4'
# output_path = 'D:/activity_surveillance/Suspecious_Video/note-giving-10.mp4'

# # Open the video file
# cap = cv2.VideoCapture(input_path)
# if not cap.isOpened():
#     print("Error: Unable to open video file.")
#     exit()

# # Get video properties
# original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Adjust frame size proportionally
# new_width = 1020
# new_height = int(original_h * (new_width / original_w))
# frame_size = (new_width, new_height)

# # Set up video writer
# video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

# # Variables for blurring logic
# blur_ratio = 50
# selected_track_id = None
# blur_all = True  # Start with all objects blurred

# # Frame processing loop
# count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error in reading frame.")
#         break

#     count += 1
#     if count % 3 != 0:  # Skip frames to reduce processing load
#         continue

#     # Resize the frame to the correct dimensions
#     frame = cv2.resize(frame, frame_size)
#     frame_original = frame.copy()  # Keep an unaltered copy for optional display

#     # Run YOLOv8 tracking on the frame
#     results = model.track(frame, persist=True, classes=0)

#     # Check if any detections exist
#     if results[0].boxes is not None and results[0].boxes.id is not None:
#         # Get boxes, class IDs, track IDs, and confidence scores
#         boxes = results[0].boxes.xyxy.int().cpu().tolist()
#         class_ids = results[0].boxes.cls.int().cpu().tolist()
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#         confidences = results[0].boxes.conf.cpu().tolist()

#         for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
#             class_name = names[class_id]
#             x1, y1, x2, y2 = box

#             # Extract the region of interest (ROI)
#             roi = frame[y1:y2, x1:x2]

#             # Apply blur if blur_all is enabled or if the track_id is not the selected one
#             if blur_all or (selected_track_id is not None and track_id != selected_track_id):
#                 blur_obj = cv2.blur(roi, (blur_ratio, blur_ratio))
#                 frame[y1:y2, x1:x2] = blur_obj  # Replace ROI with blurred region

#             # Draw bounding box
#             color = (0, 255, 0) if track_id == selected_track_id else (0, 0, 255)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             # Add text annotations
#             cvzone.putTextRect(frame, f'Track ID: {track_id}', (x1, y2 + 20), 1, 1)
#             cvzone.putTextRect(frame, f'{class_name}', (x1, y1), 1, 1)

#     # Show the processed frames
#     cv2.imshow("RGB", frame)
#     cv2.imshow("FRAME_ORIGINAL", frame_original)

#     # Write the processed frame to the output video
#     video_writer.write(frame)

#     # Key press handling
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Quit the program
#         break
#     elif key == ord('s'):  # Select a specific track_id to keep unblurred
#         try:
#             selected_track_id = int(input("Enter the track_id to keep unblurred: "))
#             blur_all = False
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")
#             selected_track_id = None
#     elif key == ord('n'):  # Blur all objects again
#         blur_all = True
#         selected_track_id = None
#     elif key == ord('+'):  # Increase blur ratio
#         blur_ratio = min(blur_ratio + 5, 100)
#         print(f"Blur ratio increased to {blur_ratio}")
#     elif key == ord('-'):  # Decrease blur ratio
#         blur_ratio = max(blur_ratio - 5, 1)
#         print(f"Blur ratio decreased to {blur_ratio}")

# # Release resources
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os

# Set environment variable to avoid OpenCV duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Callback function for mouse events (optional, can be removed if not needed)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLOv8 model
model = YOLO("yolo11x.pt")
names = model.model.names  # Get class names

# Video input/output paths
input_path = 'D:/activity_surveillance/action_class/note_giving/note-giving-128.mp4'
output_path = 'D:/activity_surveillance/Suspecious_Video/note-giving-128.mp4'

# Open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Adjust frame size proportionally
new_width = 1020
new_height = int(original_h * (new_width / original_w))
frame_size = (new_width, new_height)

# Set up video writer
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

# Variables for blurring logic
blur_ratio = 50
unblurred_track_ids = []  # List to store track IDs to keep unblurred
blur_all = False  # Blurring is disabled by default

# Frame processing loop
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error in reading frame.")
        break

    count += 1
    if count % 1 != 0:  # Skip frames to reduce processing load
        continue

    # Resize the frame to the correct dimensions
    frame = cv2.resize(frame, frame_size)
    frame_original = frame.copy()  # Keep an unaltered copy for optional display

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, classes=0)

    # Check if any detections exist
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get boxes, class IDs, track IDs, and confidence scores
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            class_name = names[class_id]
            x1, y1, x2, y2 = box

            # Extract the region of interest (ROI)
            roi = frame[y1:y2, x1:x2]

            # Apply blur if blur_all is enabled or the track_id is not in the unblurred list
            if blur_all or (track_id not in unblurred_track_ids):
                blur_obj = cv2.blur(roi, (blur_ratio, blur_ratio))
                frame[y1:y2, x1:x2] = blur_obj  # Replace ROI with blurred region

            # Draw bounding box
            color = (0, 255, 0) if track_id in unblurred_track_ids else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add text annotations
            cvzone.putTextRect(frame, f'Track ID: {track_id}', (x1, y2 + 20), 1, 1)
            cvzone.putTextRect(frame, f'{class_name}', (x1, y1), 1, 1)

    # Show the processed frames
    cv2.imshow("RGB", frame)
    cv2.imshow("FRAME_ORIGINAL", frame_original)

    # Write the processed frame to the output video
    video_writer.write(frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('s'):  # Add a track ID to keep unblurred
        try:
            track_id = int(input("Enter the track_id to keep unblurred: "))
            if track_id not in unblurred_track_ids:
                unblurred_track_ids.append(track_id)  # Add the ID to the list
                print(f"Track ID {track_id} added to unblurred list.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    elif key == ord('r'):  # Remove a track ID from the unblurred list
        try:
            track_id = int(input("Enter the track_id to remove from unblurred: "))
            if track_id in unblurred_track_ids:
                unblurred_track_ids.remove(track_id)  # Remove the ID from the list
                print(f"Track ID {track_id} removed from unblurred list.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    elif key == ord('n'):  # Blur all objects again
        blur_all = True
        unblurred_track_ids.clear()  # Clear the list of unblurred IDs
        print("Blurring all objects.")
    elif key == ord('+'):  # Increase blur ratio
        blur_ratio = min(blur_ratio + 5, 100)
        print(f"Blur ratio increased to {blur_ratio}")
    elif key == ord('-'):  # Decrease blur ratio
        blur_ratio = max(blur_ratio - 5, 1)
        print(f"Blur ratio decreased to {blur_ratio}")

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
