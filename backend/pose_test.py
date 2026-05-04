import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/pose_landmarker_lite.task"
VIDEO_PATH = "squat_sample.mp4"

#Calculate angles
def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    
    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by
    # Calculate the angle using the dot product formula
    dot_product = ba_x * bc_x + ba_y * bc_y
    magnitude_ba = math.sqrt(ba_x**2 + ba_y**2)
    magnitude_bc = math.sqrt(bc_x**2 + bc_y**2)
    
    # To avoid division by zero, check if the magnitudes are non-zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0
    # Clamp the cosine value to the range [-1, 1] to avoid numerical issues with acos
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(min(cosine_angle, 1), -1)
    # Calculate the angle in degrees
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Helper function to draw landmarks and connections on the frame
def get_landmark_points(landmarks, landmark_index, width, height):
    landmark = landmarks[landmark_index]
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)


def draw_landmarks_on_frame(frame, detection_result):
    """
    Draw pose landmarks on the frame.

    MediaPipe gives normalized coordinates:
    x and y are between 0 and 1.
    We multiply by frame width and height to get pixel positions.
    """

    if not detection_result.pose_landmarks:
        return frame, None

    height, width, _ = frame.shape

    # First detected person
    pose_landmarks = detection_result.pose_landmarks[0]

    # Draw landmark dots
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Draw simple skeleton connections
    connections = [
        # shoulders / arms
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),

        # torso
        (11, 23),
        (12, 24),
        (23, 24),

        # legs
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]

    for start_idx, end_idx in connections:
        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]

        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))

        cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    return frame, pose_landmarks


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        print(f"Check that {VIDEO_PATH} is inside your backend folder.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as pose_landmarker:
        frame_index = 0
        
        rep_count = 0
        squat_position = "up"  # Start in the "up" position

        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp_ms = int((frame_index / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # IMPORTANT:
            # Because running_mode is VIDEO, we use detect_for_video.
            detection_result = pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            frame, pose_landmarks = draw_landmarks_on_frame(frame, detection_result)
            
            if pose_landmarks:
                height, width, _ = frame.shape
                # Get the coordinates of the relevant landmarks
                
                left_hip = get_landmark_points(pose_landmarks, 23, width, height)
                left_knee = get_landmark_points(pose_landmarks, 25, width, height)
                left_ankle = get_landmark_points(pose_landmarks, 27, width, height)
                # Calculate the angle at the left knee
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                if left_knee_angle < 100 and squat_position == "up":
                    squat_position = "down"
                
                if left_knee_angle > 150 and squat_position == "down":
                    squat_position = "up"
                    rep_count += 1
                    
                # Display the angle on the frame
                
                
                label_angle = f"{int(left_knee_angle)} deg"
                label_reps = f"Reps: {rep_count}"
                label_position = f"Position: {squat_position}"
                
                cv2.rectangle(
                    frame,
                    (20, 20),
                    (300, 70),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label_angle,
                    (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label_reps,
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label_position,
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("LiftLens Pose Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()