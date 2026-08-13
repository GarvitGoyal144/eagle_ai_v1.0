import sys
import traceback
import shutil

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config.settings import settings

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save the uploaded video to disk for later processing."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    settings.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    dest = settings.UPLOAD_FOLDER / file.filename

    try:
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video: {exc}",
        )
    finally:
        await file.close()

    return {
        "status": "success",
        "filename": file.filename,
        "path": str(dest),
        "message": "Video saved. Processing engine initializing...",
    }


from pydantic import BaseModel

class ProcessRequest(BaseModel):
    filename: str

@router.post("/process")
def process_video(req: ProcessRequest):
    """Run YOLO and MobileCLIP on the uploaded video."""
    print(f"📥 /video/process called for: {req.filename}", flush=True)
    
    video_path = settings.UPLOAD_FOLDER / req.filename
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}", flush=True)
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        from app.services.vision.video_processor import video_processor
        print(f"🚀 Starting video_processor.process()...", flush=True)
        insights = video_processor.process(str(video_path), req.filename)
        print(f"✅ video_processor.process() returned successfully", flush=True)
        return insights
    except Exception as e:
        # Print the FULL traceback to Render logs
        tb = traceback.format_exc()
        print(f"❌ /video/process CRASHED with {type(e).__name__}:", flush=True)
        print(tb, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
