from datetime import datetime, timezone


class EventEngine:
    """
    Converts raw detections into meaningful surveillance events.
    """

    def __init__(self):
        self.active_tracks = {}
        self.source_name = "webcam"  # updated per session by camera or video processor

    def set_source(self, source: str):
        """Set the camera/video source name for event labeling."""
        self.source_name = source

    def process(self, detections):

        events = []

        current_tracks = set()

        now = datetime.now(timezone.utc)

        for detection in detections:

            track_id = detection["track_id"]

            current_tracks.add(track_id)

            if track_id not in self.active_tracks:

                self.active_tracks[track_id] = now

                events.append({

                    "event_type": "PERSON_ENTERED"
                    if detection["class_name"] == "person"
                    else "OBJECT_DETECTED",

                    "track_id": track_id,

                    "class_name": detection["class_name"],

                    "confidence": detection["confidence"],

                    "bbox": detection["bbox"],

                    "camera": self.source_name,

                    "timestamp": now

                })

        lost_tracks = []

        for track_id in self.active_tracks:

            if track_id not in current_tracks:

                events.append({

                    "event_type": "TRACK_LOST",

                    "track_id": track_id,

                    "camera": self.source_name,

                    "timestamp": now

                })

                lost_tracks.append(track_id)

        for track_id in lost_tracks:

            del self.active_tracks[track_id]

        return events


event_engine = EventEngine()