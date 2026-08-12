import asyncio
import cv2
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config.settings import settings
from app.services.camera.camera_manager import camera_manager
from app.services.vision.annotator import draw_detections
from app.services.vision.inference_worker import inference_worker

router = APIRouter(prefix="/camera", tags=["camera"])


@router.post("/start")
def start_camera():
    if camera_manager.start():
        return {"success": True, "message": "Camera started"}
    raise HTTPException(status_code=500, detail="Unable to open camera hardware")


@router.post("/stop")
def stop_camera():
    camera_manager.stop()
    return {"success": True, "message": "Camera stopped"}


@router.get("/status")
def camera_status():
    return {
        "is_running": camera_manager.is_running,
        "ai_enabled": camera_manager.ai_enabled,
    }


@router.post("/ai/toggle")
def toggle_ai():
    """Toggle AI features (detection, tracking, annotations) on or off."""
    new_state = not camera_manager.ai_enabled
    camera_manager.set_ai_enabled(new_state)
    return {
        "ai_enabled": camera_manager.ai_enabled,
        "message": "AI features enabled" if new_state else "AI features disabled — streaming raw feed",
    }


def generate_frames():
    """
    MJPEG stream generator.

    When AI is enabled:  submit frames to inference, draw bounding boxes.
    When AI is disabled: stream raw camera frames with zero processing overhead.
    """
    try:
        while True:
            if not camera_manager.is_running:
                time.sleep(0.1)
                continue

            frame = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if camera_manager.ai_enabled:
                # AI mode — run inference + draw annotations
                inference_worker.submit_frame(frame)
                display = draw_detections(frame, inference_worker.get_detections())
            else:
                # Raw mode — pass through untouched
                display = frame

            ret, buffer = cv2.imencode(
                ".jpg",
                display,
                [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY],
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    except (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt):
        pass


@router.get("/live")
def live_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
