import sys
import traceback
import shutil
import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

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
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}")
    finally:
        await file.close()

    return {
        "status": "success",
        "filename": file.filename,
        "path": str(dest),
        "message": "Video saved.",
    }


class ProcessRequest(BaseModel):
    filename: str


def _run_clip_embeddings_background(video_path: str, filename: str, session_id: str):
    """Run CLIP scene embedding in background after YOLO phase completes."""
    try:
        import cv2
        import time
        from app.services.embeddings.embedding_engine import embedding_engine
        from app.services.event_service import event_service

        print(f"🧠 [CLIP BG] Starting background semantic embedding for {filename}...", flush=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample 1 scene snapshot every 5 seconds
        scene_interval_frames = int(fps * 5.0)
        snapshots = 0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % scene_interval_frames == 0:
                try:
                    timestamp_sec = round(frame_idx / fps, 2)
                    res = embedding_engine.classify_scene_caption(frame)
                    event_service.save_scene_embedding(
                        embedding=res["embedding"],
                        timestamp=time.time(),
                        caption=res.get("caption", ""),
                        category=res.get("category", "normal"),
                        camera=filename,
                        snapshot_id=f"scene_{session_id}_{frame_idx}",
                        frame_number=frame_idx,
                        timestamp_sec=timestamp_sec,
                        video_filename=filename,
                        session_id=session_id,
                    )
                    snapshots += 1
                    del frame
                except Exception as e:
                    print(f"⚠️ [CLIP BG] Scene encode error at frame {frame_idx}: {e}", flush=True)
                    del frame
            else:
                del frame
            frame_idx += 1

        cap.release()
        print(f"✅ [CLIP BG] Semantic embedding complete — {snapshots} scene snapshots saved for {filename}", flush=True)
    except Exception as e:
        print(f"❌ [CLIP BG] Background embedding failed: {e}", flush=True)
        traceback.print_exc()


@router.post("/process")
def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Phase 1: Run YOLO object detection locally (fast, <15s on Render CPU).
    Phase 2: Run CLIP semantic scene embedding in the background after response is returned.
    """
    print(f"📥 /video/process called for: {req.filename}", flush=True)

    video_path = settings.UPLOAD_FOLDER / req.filename
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}", flush=True)
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        from app.services.vision.video_processor import video_processor
        print(f"🚀 Starting YOLO detection phase...", flush=True)
        insights = video_processor.process(str(video_path), req.filename)
        print(f"✅ YOLO phase complete. Scheduling semantic embedding...", flush=True)

        # Schedule CLIP background embedding — happens after we respond to the browser
        if not settings.DISABLE_CLIP:
            background_tasks.add_task(
                _run_clip_embeddings_background,
                str(video_path),
                req.filename,
                insights.get("session_id", ""),
            )

        return insights

    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ /video/process CRASHED with {type(e).__name__}:", flush=True)
        print(tb, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
