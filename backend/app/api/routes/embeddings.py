import cv2
from fastapi import APIRouter, HTTPException

from app.services.camera.camera_manager import camera_manager
from app.services.embeddings.embedding_engine import embedding_engine

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"]
)


@router.get("/test")
def test_embedding():
    """
    Generate a test CLIP embedding from the currently running camera.

    Uses the shared CameraManager instead of opening a second
    VideoCapture — avoids hardware conflicts on single-camera
    systems.
    """

    if not camera_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Camera is not running. Start the camera first via /camera/start",
        )

    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(
            status_code=500,
            detail="Camera is running but failed to capture a frame",
        )

    embedding = embedding_engine.encode_frame(frame)

    return {
        "dimension": len(embedding),
        "sample": embedding[:5].tolist()
    }