import cv2
import mediapipe as mp

def main():
    video_path = "sample_squat.mp4"     # Replace with your video path
    
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
            
            rgb_frame = cv2.