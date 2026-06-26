import json
from pathlib import Path

from analysis_engine import analyse_video


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "squat_sample.mp4"
OUTPUT_PATH = BASE_DIR / "analysis_result.json"


def main():
    print("Running LiftLens analysis...")

    summary = analyse_video(VIDEO_PATH)

    print("\n=== LiftLens Analysis Summary ===")
    print(json.dumps(summary, indent=4))

    with open(OUTPUT_PATH, "w") as file:
        json.dump(summary, file, indent=4)

    print(f"\nAnalysis saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()