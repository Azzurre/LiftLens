import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    video_path = "squat_sample.mp4"     # Replace with your video path
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        
        while True:
            success, frame = cap.read()
            
            if not success:
                print("End of video.")
                break
            
            # Convert the BGR image to RGB by OpenCV for MediaPipe
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)
            
            # Draw landmarks if pose is detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                
                cv2.imshow("LiftLens Pose Test", frame)
                
                # Press q to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()