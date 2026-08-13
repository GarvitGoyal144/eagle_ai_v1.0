import os
import sys
import cv2
import tempfile
import traceback
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response

from app.config.settings import settings
from app.database.mongodb import mongodb

router = APIRouter(prefix="/visual", tags=["Visual Retrieval"])


def get_event_metadata(event_id: str):
    if not mongodb.database:
        raise HTTPException(status_code=500, detail="Database not connected")

    event = mongodb.database.events.find_one({"event_id": event_id})
    if not event:
        event = mongodb.database.scene_embeddings.find_one({"snapshot_id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event or Scene not found")

    return event


@router.get("/frame/{event_id}")
def get_frame(event_id: str):
    """Return a base64 encoded JPEG or image/jpeg response for a specific event."""
    try:
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

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Could not read frame from video")

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode frame to JPEG")

        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /visual/frame error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clip/{event_id}")
def get_clip(event_id: str, background_tasks: BackgroundTasks):
    """Generate and stream a 5-second MP4 clip around the event timestamp."""
    try:
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
            raise HTTPException(status_code=500, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 5 seconds total (2.5s before, 2.5s after)
        clip_frames = int(fps * 5)
        start_frame = max(0, int(frame_number) - int(fps * 2.5))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create temporary MP4 file
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # Try browser-compatible codecs
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*"avc1"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cv2.VideoWriter_fourcc(*"MJPG"),
        ]

        out = None
        for fourcc in codecs_to_try:
            try:
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
            except Exception:
                continue

        if not out or not out.isOpened():
            cap.release()
            # Fallback: if VideoWriter fails on headless Linux, redirect to the exact JPEG frame
            return get_frame(event_id)

        frames_written = 0
        while frames_written < clip_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1

        out.release()
        cap.release()

        if frames_written == 0 or not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return get_frame(event_id)

        def cleanup(path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        background_tasks.add_task(cleanup, temp_path)
        return FileResponse(temp_path, media_type="video/mp4", filename=f"clip_{event_id[:8]}.mp4")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /visual/clip error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
