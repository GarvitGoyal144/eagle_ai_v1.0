from datetime import datetime
import uuid

from app.database.mongodb import mongodb


class EventService:
    """
    Stores unique tracking events in MongoDB.
    """

    def __init__(self):
        self.active_tracks = set()

    def save_new_tracks(self, detections):

        if mongodb.database is None:
            return

        for detection in detections:

            track_id = detection["track_id"]

            if track_id in self.active_tracks:
                continue

            self.active_tracks.add(track_id)

            document = {
                "event_id": str(uuid.uuid4()),
                "track_id": track_id,
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "camera": "webcam",
                "timestamp": datetime.utcnow(),
            }

            mongodb.database.events.insert_one(document)

            print(f"✅ New Event Saved: {detection['class_name']} #{track_id}")


event_service = EventService()