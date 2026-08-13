import os

# Disable Ultralytics hub/update checks to prevent loading hangs in offline/firewalled cloud environments
os.environ["ULTRALYTICS_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "False"

import torch
from ultralytics import YOLO

from app.config.settings import settings


class DetectionService:
    """
    Detection + tracking using YOLO26 with OC-SORT tracker.
    OC-SORT provides superior occlusion recovery and trajectory estimation
    at real-time FPS without heavy Re-ID neural network overhead.
    """

    def __init__(self):
        self.model = None
        self.tracker = settings.TRACKER or "ocsort.yaml"

        if settings.DEVICE.lower() == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif settings.DEVICE.lower() == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self.model is not None:
            return

        model_path = settings.YOLO_MODEL or "yolo26n.pt"
        print(f"Loading YOLO26 ({model_path}) on {self.device.upper()}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"YOLO26 loaded ✅  tracker: {self.tracker}")

    def track(self, frame):
        self._ensure_loaded()

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            device=self.device,
            conf=settings.DETECTION_CONF,
            imgsz=settings.INFERENCE_SIZE,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):
                detections.append(
                    {
                        "track_id": int(track_id),
                        "class_id": int(box.cls),
                        "class_name": self.model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

        return detections


detection_service = DetectionService()
