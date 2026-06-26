import cv2
import mediapipe as mp
import math
import json
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

def landmarks_are_visible(landmarks, indices, min_visibility=0.5):
    """
    Check if the specified landmarks are visible based on their visibility scores.
    """
    for index in indices:
        if landmarks[index].visibility < min_visibility:
            return False
    return True

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

def classify_squat_position(min_knee_angle):
    """
    Classifies squat depth based on the lowest knee angle reached during a repetition.
    Lower knee angle generally means deeper squat.
    These thresholds are beginner friendly estimates and can be tuned later.
    """
    
    if min_knee_angle is None:
        return "No depth detected"
    
    if min_knee_angle < 90:
        return "Good depth"
    elif min_knee_angle < 115:
        return "Almost deep enough"
    else:
        return "Too shallow"


def generate_analysis_summary(rep_count, rep_depths):
    """
    Generates a summary of the squat analysis.
    """
    if rep_count == 0:
        return {
            "reps": 0,
            "average_depth": None,
            "best_depth": None,
            "score": 0,
            "feedback": [
                "No full squat reps were detected",
                "Make sure your full body is visible and try again"
            ]
        }
    
    average_depth = sum(rep_depths) / len(rep_depths) if rep_depths else None
    best_depth = min(rep_depths) if rep_depths else None
    
    good_reps = 0
    shallow_reps = 0
    
    for depth in rep_depths:
        if depth < 90:
            good_reps += 1
        elif depth < 115:
            shallow_reps += 1
    feedback = []
    if good_reps == rep_count:
        feedback.append("All reps were good depth!")
    elif good_reps > rep_count / 2:
        feedback.append(f"{good_reps} out of {rep_count} reps were good depth.")
        
    if shallow_reps > 0:
        feedback.append(f"{shallow_reps} reps were too shallow. Try to go lower.")
        
    if average_depth is not None:
        if average_depth < 90:
            feedback.append("Your average squat depth is good.")
        elif average_depth < 115:
            feedback.append("Your average squat depth is almost deep enough.")
        else:
            feedback.append("Your average squat depth is too shallow. Try to go lower.")
    
    score = 100
    
    if average_depth is not None:
        if average_depth < 90:
            score -= int((average_depth - 90) * 1.5)
    score -= shallow_reps * 8
    
    score = max(0, min(100, score))
    
    return {
        "reps": rep_count,
        "average_depth": round(average_depth, 2) if average_depth is not None else None,
        "best_depth": round(best_depth, 2) if best_depth is not None else None,
        "score": score,
        "feedback": feedback,
        "rep_depths": [round(depth, 2) for depth in rep_depths]
    }
    

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
        
        down_frames = 0
        up_frames = 0
        
        MIN_DOWN_FRAMES = 5
        MIN_UP_FRAMES = 5
        
        rep_cooldown = 0
        COOLDOWN_FRAMES = 10
        
        current_rep_min_angle = None
        rep_depths = []
        rep_details = []
        last_depth_feedback = "No reps yet"

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
                
                left_leg_landmarks = [23, 25, 27]

                if not landmarks_are_visible(pose_landmarks, left_leg_landmarks):
                    cv2.putText(
                        frame,
                        "Pose unclear - adjust camera",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("LiftLens Pose Test", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    frame_index += 1
                    continue
                
                left_hip = get_landmark_points(pose_landmarks, 23, width, height)
                left_knee = get_landmark_points(pose_landmarks, 25, width, height)
                left_ankle = get_landmark_points(pose_landmarks, 27, width, height)
                # Calculate the angle at the left knee
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                DOWN_ANGLE = 100
                UP_ANGLE = 150
                
                if rep_cooldown > 0:
                    rep_cooldown -= 1
                    
                    
                if left_knee_angle < DOWN_ANGLE:
                    down_frames += 1
                else:
                    down_frames = 0

                if left_knee_angle > UP_ANGLE :
                    up_frames += 1
                else:
                    up_frames = 0

                if down_frames >= MIN_DOWN_FRAMES and squat_position == "up":
                    squat_position = "down"
                    current_rep_min_angle = left_knee_angle  # Start tracking the minimum angle for this rep
                    if squat_position == "down":
                        if current_rep_min_angle is None or left_knee_angle < current_rep_min_angle:
                            current_rep_min_angle = left_knee_angle
                    
                if (up_frames >= MIN_UP_FRAMES and squat_position == "down" and rep_cooldown == 0):
                    squat_position = "up"
                    rep_count += 1
                    rep_cooldown = COOLDOWN_FRAMES
                    
                    if current_rep_min_angle is not None:
                        rep_depths.append(current_rep_min_angle)
                        last_depth_feedback = classify_squat_position(current_rep_min_angle)
                        rep_details.append({
                            "rep_number": rep_count,
                            "depth_angle": round(current_rep_min_angle, 2),
                            "depth_feedback": last_depth_feedback
                        })
                    current_rep_min_angle = None  # Reset for the next rep
                # Display the angle on the frame
                
                if rep_depths:
                    average_depth = sum(rep_depths) / len(rep_depths)
                    best_depth = min(rep_depths)
                else:
                    average_depth = None
                    best_depth = None
                    
                if current_rep_min_angle is not None:
                    label_current_depth = f"Current rep depth: {int(current_rep_min_angle)} deg"
                else:
                    label_current_depth = "Current rep depth: N/A"
                
                if rep_depths:
                    label_last_depth = f"Last rep depth: {int(rep_depths[-1])} deg ({last_depth_feedback})"
                    label_average_depth = f"Average depth: {int(average_depth)} deg"
                    label_best_depth = f"Best depth: {int(best_depth)} deg"
                else:
                    label_last_depth = "Last rep depth: N/A"
                    label_average_depth = "Average depth: N/A"
                    label_best_depth = "Best depth: N/A"
                
                
                label_angle = f"{int(left_knee_angle)} deg"
                label_reps = f"Reps: {rep_count}"
                label_position = f"Position: {squat_position}"
                
                
                label_down_frames = f"Down frames: {down_frames}"
                label_up_frames = f"Up frames: {up_frames}"
                label_cooldown = f"Cooldown: {rep_cooldown}"
                cv2.rectangle(
                    frame,
                    (20, 20),
                    (460, 230),
                    (0, 0, 0),
                    -1,
                )
                
                cv2.putText(frame, label_angle, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label_reps, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label_position, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(frame, label_current_depth, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label_last_depth, (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label_average_depth, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label_best_depth, (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("LiftLens Pose Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    summary = generate_analysis_summary(rep_count, rep_depths)
    
    print("\n === LiftLens Analysis Summary ===")
    print(json.dumps(summary, indent=4))
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()