import cv2
import mediapipe as mp

def main():
    video_path = "sample_squat.mp4"     # Replace with your video path
    
    cap = cv2.VideoCapture(video_path)
    