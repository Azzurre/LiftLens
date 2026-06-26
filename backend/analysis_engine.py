import os

# Reduce some noisy logs from MediaPipe/TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import cv2
import mediapipe as mp
import math
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# This makes the model path work even if you run Python from another folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b.

    For squat knee angle:
    a = hip
    b = knee
    c = ankle
    """

    ax, ay = a
    bx, by = b
    cx, cy = c

    # Create vectors from the knee to the hip and ankle
    ba_x = ax - bx
    ba_y = ay - by

    bc_x = cx - bx
    bc_y = cy - by

    # Dot product
    dot_product = (ba_x * bc_x) + (ba_y * bc_y)

    # Vector lengths
    magnitude_ba = math.sqrt((ba_x ** 2) + (ba_y ** 2))
    magnitude_bc = math.sqrt((bc_x ** 2) + (bc_y ** 2))

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Prevent tiny floating point errors from crashing acos()
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.degrees(math.acos(cosine_angle))

    return angle


def get_landmark_point(landmarks, landmark_index, width, height):
    """
    Convert MediaPipe normalised coordinates into pixel coordinates.
    """

    landmark = landmarks[landmark_index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)


def landmarks_are_visible(landmarks, landmark_indices, min_visibility=0.5):
    """
    Checks if important landmarks are visible enough to trust.

    Example:
    For the left leg, we check:
    left hip, left knee, left ankle.
    """

    for index in landmark_indices:
        landmark = landmarks[index]

        # Some versions expose visibility. If it does not exist, we do not block.
        visibility = getattr(landmark, "visibility", 1.0)

        if visibility < min_visibility:
            return False

    return True


def classify_squat_depth(min_knee_angle):
    """
    Classify squat depth based on the lowest knee angle reached during a rep.

    Lower knee angle = deeper squat.
    These are beginner-friendly starter thresholds.
    """

    if min_knee_angle is None:
        return "No depth detected"

    if min_knee_angle <= 90:
        return "Good depth"
    elif min_knee_angle <= 110:
        return "Almost deep enough"
    else:
        return "Too shallow"


def generate_analysis_summary(rep_count, rep_depths, rep_details):
    """
    Creates the final analysis result.

    This is the data that our FastAPI backend will later return to React.
    """

    if rep_count == 0:
        return {
            "reps": 0,
            "average_depth": None,
            "best_depth": None,
            "score": 0,
            "feedback": [
                "No full squat reps were detected.",
                "Make sure your full body is visible and try again."
            ],
            "rep_depths": [],
            "rep_details": []
        }

    average_depth = sum(rep_depths) / len(rep_depths) if rep_depths else None
    best_depth = min(rep_depths) if rep_depths else None

    good_reps = 0
    shallow_reps = 0

    for depth in rep_depths:
        if depth <= 90:
            good_reps += 1
        elif depth > 110:
            shallow_reps += 1

    feedback = []

    if good_reps == rep_count:
        feedback.append("Great depth on all detected reps.")
    elif good_reps >= rep_count / 2:
        feedback.append("Good depth on most reps.")
    else:
        feedback.append("You may need to squat deeper on more reps.")

    if shallow_reps > 0:
        feedback.append(f"{shallow_reps} rep(s) looked too shallow.")

    if average_depth is not None:
        if average_depth <= 90:
            feedback.append("Your average squat depth looks strong.")
        elif average_depth <= 110:
            feedback.append("Your average depth is close, but could improve slightly.")
        else:
            feedback.append("Your average depth suggests the squats may be too shallow.")

    # Simple starter scoring system
    score = 100

    if average_depth is not None and average_depth > 90:
        score -= int((average_depth - 90) * 1.5)

    score -= shallow_reps * 8
    score = max(0, min(100, score))

    return {
        "reps": rep_count,
        "average_depth": round(average_depth, 2) if average_depth is not None else None,
        "best_depth": round(best_depth, 2) if best_depth is not None else None,
        "score": score,
        "feedback": feedback,
        "rep_depths": [round(depth, 2) for depth in rep_depths],
        "rep_details": rep_details
    }


def analyse_video(video_path):
    """
    Main reusable analysis function.

    Input:
        video_path: path to squat video

    Output:
        summary dictionary with reps, depth, score, and feedback
    """

    video_path = str(video_path)

    if not MODEL_PATH.exists():
        return {
            "error": "Pose model file not found.",
            "expected_model_path": str(MODEL_PATH)
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "error": "Could not open video.",
            "video_path": video_path
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Rep counting state
    rep_count = 0
    squat_position = "up"

    down_frames = 0
    up_frames = 0

    MIN_DOWN_FRAMES = 5
    MIN_UP_FRAMES = 5

    rep_cooldown = 0
    COOLDOWN_FRAMES = 10

    DOWN_ANGLE = 100
    UP_ANGLE = 150

    # Depth tracking
    current_rep_min_angle = None
    rep_depths = []
    rep_details = []

    frame_index = 0
    skipped_unclear_frames = 0
    processed_pose_frames = 0

    with vision.PoseLandmarker.create_from_options(options) as pose_landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp_ms = int((frame_index / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.copy()

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            detection_result = pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if detection_result.pose_landmarks:
                height, width, _ = frame.shape
                pose_landmarks = detection_result.pose_landmarks[0]

                # Left leg landmarks:
                # 23 = left hip
                # 25 = left knee
                # 27 = left ankle
                left_leg_indices = [23, 25, 27]

                if not landmarks_are_visible(pose_landmarks, left_leg_indices):
                    skipped_unclear_frames += 1
                    frame_index += 1
                    continue

                processed_pose_frames += 1

                left_hip = get_landmark_point(pose_landmarks, 23, width, height)
                left_knee = get_landmark_point(pose_landmarks, 25, width, height)
                left_ankle = get_landmark_point(pose_landmarks, 27, width, height)

                left_knee_angle = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle,
                )

                # Cooldown prevents duplicate fake reps
                if rep_cooldown > 0:
                    rep_cooldown -= 1

                # Stable down detection
                if left_knee_angle < DOWN_ANGLE:
                    down_frames += 1
                else:
                    down_frames = 0

                # Stable up detection
                if left_knee_angle > UP_ANGLE:
                    up_frames += 1
                else:
                    up_frames = 0

                # Confirm the person is down
                if down_frames >= MIN_DOWN_FRAMES and squat_position == "up":
                    squat_position = "down"
                    current_rep_min_angle = left_knee_angle

                # While down, track the deepest point
                if squat_position == "down":
                    if (
                        current_rep_min_angle is None
                        or left_knee_angle < current_rep_min_angle
                    ):
                        current_rep_min_angle = left_knee_angle

                # Count a rep when the person returns up
                if (
                    up_frames >= MIN_UP_FRAMES
                    and squat_position == "down"
                    and rep_cooldown == 0
                ):
                    squat_position = "up"
                    rep_count += 1
                    rep_cooldown = COOLDOWN_FRAMES

                    if current_rep_min_angle is not None:
                        rep_depths.append(current_rep_min_angle)

                        depth_feedback = classify_squat_depth(
                            current_rep_min_angle
                        )

                        rep_details.append({
                            "rep_number": rep_count,
                            "depth_angle": round(current_rep_min_angle, 2),
                            "depth_feedback": depth_feedback
                        })

                    current_rep_min_angle = None

            frame_index += 1

    cap.release()

    summary = generate_analysis_summary(
        rep_count,
        rep_depths,
        rep_details,
    )

    summary["debug"] = {
        "total_frames": frame_index,
        "processed_pose_frames": processed_pose_frames,
        "skipped_unclear_frames": skipped_unclear_frames
    }

    return summary