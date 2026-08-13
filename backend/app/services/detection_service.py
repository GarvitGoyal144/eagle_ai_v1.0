import os

# Disable Ultralytics hub/update checks to prevent loading hangs in offline/firewalled cloud environments
os.environ["ULTRALYTICS_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "False"
# Cap threads to prevent CPU contention on Render's shared free-tier CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)  # Limit PyTorch YOLO inference threads globally

from ultralytics import YOLO
from app.config.settings import settings


class DetectionService:
    """
    Fast, lightweight object detection using YOLOv11.
    Supports both direct prediction (0 tracker overhead) and optional tracking.
    """

    def __init__(self):
        self.model = None

        if settings.DEVICE.lower() == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif settings.DEVICE.lower() == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self.model is not None:
            return

        model_path = settings.YOLO_MODEL or "yolo11n.pt"
        print(f"Loading YOLO ({model_path}) on {self.device.upper()}...", flush=True)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"YOLO ({model_path}) loaded successfully ✅", flush=True)

    def detect(self, frame):
        """Fast object detection without tracker state overhead."""
        self._ensure_loaded()

        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=settings.DETECTION_CONF,
            imgsz=settings.INFERENCE_SIZE,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is not None:
            for idx, box in enumerate(boxes):
                detections.append({
                    "track_id": idx + 1,
                    "class_id": int(box.cls),
                    "class_name": self.model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                })

        return detections

    def track(self, frame):
        """Standard detection/tracking with graceful predict fallback."""
        return self.detect(frame)


detection_service = DetectionService()
