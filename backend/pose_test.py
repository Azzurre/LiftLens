import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/pose_landmarker_lite.task"
VIDEO_PATH = "squat_sample.mp4"     # Replace with your video path


def draw_landmarks_on_frame(frame, detection_result):
    """Draw pose landmarks on the frame.
    
    MediaPipe provides normalized coordinates, so we need to convert them to pixel coordinates.
    """
    
    if not detection_result.pose_landmarks:
        return frame
    
    height, width, _ = frame.shape
    
    pose_landmarks = detection_result.pose_landmarks[0]  # Get the first detected pose
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle for each landmark
        
    #draw connections between landmarks
    
    connections = [
        #arms Shoulders
        (11, 12)
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (14, 16),
        
        #torso
        (11, 23),
        (12, 24),
        (23, 24),
        
        #legs
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28)
    ]
    
    for start_idx, end_idx in connections:
        start= pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]
        
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Draw a blue line for connections
    
def main():
    video_path = VIDEO_PATH     # Replace with your video path
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_tracking_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(options) as pose_landmarker:
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_index = 0
        while True:
            success, frame = cap.read()
            
            if not success:
                break
            
            timestamp_ms = int((frame_index / fps) * 1000)
            
            # Convert the frame to RGB as MediaPipe expects RGB input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #create a MediaPipe Image from the RGB frame
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, 
                data=rgb_frame
            )
            
            # Process the frame with MediaPipe Pose
            detection_result = pose_landmarker.detect(mp_image, timestamp_ms)
            
            # Draw landmarks on the frame
            draw_landmarks_on_frame(frame, detection_result)
            
            # Display the frame
            cv2.imshow("LiftLens Pose Test", frame)
            
            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()