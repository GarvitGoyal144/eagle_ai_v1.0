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


@router.get("/progress/{filename}")
def get_video_progress(filename: str):
    """Return real-time progress percentage, frame counters, and detections for a video."""
    from app.services.vision.video_processor import progress_store
    prog = progress_store.get(filename)
    if not prog:
        return {
            "status": "idle",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "detections": 0,
            "step": "Ready for analysis",
        }
    return prog


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
        # Sample 4 strategic keyframes across the video (10%, 35%, 65%, 90%)
        snapshot_frames = [
            int(total_frames * 0.1),
            int(total_frames * 0.35),
            int(total_frames * 0.65),
            max(0, int(total_frames * 0.9)),
        ]
        snapshots = 0

        for frame_idx in snapshot_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
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

        cap.release()
        print(f"✅ [CLIP BG] Semantic embedding complete — {snapshots} scene snapshots saved for {filename}", flush=True)
    except Exception as e:
        print(f"❌ [CLIP BG] Background embedding failed: {e}", flush=True)
        traceback.print_exc()


@router.post("/process")
def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Immediately returns a session_id and starts YOLO + CLIP processing
    in the background. Frontend polls /video/progress/{filename} for status
    and reads insights from there when status == 'completed'.
    """
    print(f"📥 /video/process called for: {req.filename}", flush=True)

    video_path = settings.UPLOAD_FOLDER / req.filename
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}", flush=True)
        raise HTTPException(status_code=404, detail="Video file not found")

    import uuid
    session_id = str(uuid.uuid4())

    def _run_full_pipeline(video_path_str: str, filename: str, sid: str):
        """Run YOLO + CLIP in background — no HTTP timeout risk."""
        try:
            from app.services.vision.video_processor import video_processor
            print(f"🚀 [BG] Starting YOLO detection for {filename}...", flush=True)
            insights = video_processor.process(video_path_str, filename, sid)
            print(f"✅ [BG] YOLO complete for {filename}. Starting CLIP...", flush=True)

            if not settings.DISABLE_CLIP:
                _run_clip_embeddings_background(
                    video_path_str,
                    filename,
                    insights.get("session_id", sid),
                )
        except Exception as e:
            print(f"❌ [BG] Pipeline failed for {filename}: {e}", flush=True)
            traceback.print_exc()
            from app.services.vision.video_processor import progress_store
            progress_store[filename] = {
                "status": "error",
                "progress": 0,
                "step": f"Processing failed: {e}",
            }

    background_tasks.add_task(
        _run_full_pipeline,
        str(video_path),
        req.filename,
        session_id,
    )

    print(f"⚡ /video/process returned immediately — pipeline running in background", flush=True)
    return {
        "status": "started",
        "filename": req.filename,
        "session_id": session_id,
        "message": "Processing started. Poll /video/progress/{filename} for updates.",
    }

