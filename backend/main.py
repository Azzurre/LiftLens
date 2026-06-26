from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from analysis_engine import analyse_video


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="LiftLens API",
    description="AI fitness form feedback backend using MediaPipe pose analysis.",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "LiftLens API is running",
        "available_endpoints": ["/analyse"],
    }


@app.post("/analyse")
async def analyse_uploaded_video(file: UploadFile = File(...)):
    allowed_extensions = [".mp4", ".mov", ".avi", ".webm", ".mkv"]

    original_filename = file.filename or "uploaded_video.mp4"
    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in allowed_extensions:
        return {
            "error": "Unsupported file type.",
            "allowed_extensions": allowed_extensions,
        }

    unique_filename = f"{uuid.uuid4()}{file_extension}"
    saved_video_path = UPLOAD_DIR / unique_filename

    try:
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        summary = analyse_video(saved_video_path)

        return {
            "filename": original_filename,
            "analysis": summary,
        }

    finally:
        file.file.close()

        if saved_video_path.exists():
            saved_video_path.unlink()