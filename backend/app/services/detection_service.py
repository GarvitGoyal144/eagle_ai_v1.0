import os

# Disable Ultralytics hub/update checks to prevent loading hangs in offline/firewalled cloud environments
os.environ["ULTRALYTICS_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "False"
# Cap threads — allow 2 for faster YOLO inference on Render's shared CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)  # Allow 2 threads for YOLO inference

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
    def detect_batch(self, frames: list) -> list:
        """Batch YOLO inference — processes all frames in one model call (much faster than per-frame)."""
        self._ensure_loaded()
        if not frames:
            return []

        all_results = self.model.predict(
            source=frames,
            device=self.device,
            conf=settings.DETECTION_CONF,
            imgsz=settings.INFERENCE_SIZE,
            verbose=False,
        )

        batch_detections = []
        for results in all_results:
            detections = []
            boxes = results.boxes
            if boxes is not None:
                for idx, box in enumerate(boxes):
                    detections.append({
                        "track_id": idx + 1,
                        "class_id": int(box.cls),
                        "class_name": self.model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    })
            batch_detections.append(detections)

        return batch_detections



detection_service = DetectionService()
