from ultralytics import YOLO
import torch


class DetectionService:
    """
    Runs object detection on incoming frames.
    """

    def __init__(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🚀 Loading YOLO11 on {device.upper()}...")

        self.model = YOLO("yolo11n.pt")

        self.model.to(device)

        self.device = device

        print("✅ YOLO Loaded")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,
            conf=0.4,
        )

        annotated = results[0].plot()

        return annotated


detection_service = DetectionService()