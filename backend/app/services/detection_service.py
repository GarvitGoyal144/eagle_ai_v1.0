from ultralytics import YOLO
import torch


class DetectionService:
    """
    Detection + Tracking Service using YOLO11 + ByteTrack.
    """

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🚀 Loading YOLO11 on {self.device.upper()}...")

        self.model = YOLO("yolo11n.pt")

        self.model.to(self.device)

        print("✅ YOLO11 Loaded")

    def detect(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            conf=0.4,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        detections = []

        boxes = results[0].boxes

        if boxes.id is not None:

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

        return annotated_frame, detections


detection_service = DetectionService()