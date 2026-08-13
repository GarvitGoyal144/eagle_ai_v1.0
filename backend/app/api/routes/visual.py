import os
import cv2
import tempfile
import base64
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.config.settings import settings
from app.database.mongodb import mongodb

router = APIRouter(prefix="/visual", tags=["Visual Retrieval"])

def get_event_metadata(event_id: str):
    if not mongodb.database:
        raise HTTPException(status_code=500, detail="Database not connected")
        
    event = mongodb.database.events.find_one({"event_id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
        
    return event

@router.get("/frame/{event_id}")
def get_frame(event_id: str):
    """Return a base64 encoded JPEG or image/jpeg response for a specific event."""
    event = get_event_metadata(event_id)
    
    filename = event.get("video_filename")
    frame_number = event.get("frame_number")
    
    if not filename or frame_number is None:
        raise HTTPException(status_code=400, detail="Event missing video metadata")
        
    video_path = settings.UPLOAD_FOLDER / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame")
        
    # Return as image/jpeg
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode frame")
        
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/clip/{event_id}")
def get_clip(event_id: str, background_tasks: BackgroundTasks):
    """Generate and stream a 5-second MP4 clip around the event timestamp."""
    event = get_event_metadata(event_id)
    
    filename = event.get("video_filename")
    frame_number = event.get("frame_number")
    
    if not filename or frame_number is None:
        raise HTTPException(status_code=400, detail="Event missing video metadata")
        
    video_path = settings.UPLOAD_FOLDER / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 5 seconds total (2.5s before, 2.5s after)
    clip_frames = int(fps * 5)
    start_frame = max(0, frame_number - int(fps * 2.5))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    frames_written = 0
    while frames_written < clip_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        
    out.release()
    cap.release()
    
    def cleanup(path):
        try:
            os.remove(path)
        except:
            pass

    # Delete temp file after streaming
    background_tasks.add_task(cleanup, temp_path)
    
    return FileResponse(temp_path, media_type="video/mp4")
