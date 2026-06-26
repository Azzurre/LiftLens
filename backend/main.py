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
    title="Video Analysis API",
    description="An API for analyzing videos using the LiftLens analysis engine.",
    version="1.0.0",
)

#React frontend runs on port 3000, so we allow CORS from that origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "LiftLens API is running",
        "available_endpoints": ["/analyse"]
    }
@app.post("/analyse")
async def analyse_uploaded_video(file: UploadFile = File(...)):
    """Receives a video file, saves it , analysis it and returns the result

    """
    
    allowed_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    
    original_filename = file.filename or "uploaded_video.mp4"
    
    file_extension = Path(original_filename).suffix.lower()
    if file_extension not in allowed_extensions:
        return {"error": "Invalid file type. Please upload a video file.",
                "allowed_extensions": allowed_extensions}
    
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    saved_video_path = UPLOAD_DIR / unique_filename
    
    try:
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        summary = analyse_video(saved_video_path)
        return {
            "filename": original_filename,
            "summary": summary
        }
        
    finally:
        file.file.close()
        if saved_video_path.exists():
            saved_video_path.unlink()  # Delete the uploaded file after processing
    