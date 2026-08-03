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
