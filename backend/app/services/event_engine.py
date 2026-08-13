from datetime import datetime, timezone


class EventEngine:
    """
    Converts raw YOLO tracking detections into structured surveillance events:
    - Object/Person Appearances (PERSON_ENTERED, OBJECT_DETECTED)
    - Track Exits (TRACK_LOST)
    
    All higher-level semantic understanding (accidents, actions, colors, context)
    is handled by MobileCLIP zero-shot vector embeddings.
    """

    def __init__(self):
        self.active_tracks = {}
        self.source_name = "webcam"

    def set_source(self, source: str):
        """Set the camera/video source name for event labeling and reset session state."""
        self.source_name = source
        self.active_tracks.clear()

    def process(self, detections):
        events = []
        current_tracks = set()
        now = datetime.now(timezone.utc)

        for detection in detections:
            track_id = detection["track_id"]
            current_tracks.add(track_id)
            cls_name = detection.get("class_name", "object")
            bbox = detection.get("bbox", [])

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "started_at": now,
                    "class_name": cls_name,
                    "last_bbox": bbox,
                }

                events.append({
                    "event_type": "PERSON_ENTERED"
                    if cls_name == "person"
                    else "OBJECT_DETECTED",
                    "track_id": track_id,
                    "class_name": cls_name,
                    "confidence": detection.get("confidence", 0.0),
                    "bbox": bbox,
                    "camera": self.source_name,
                    "timestamp": now,
                })
            else:
                self.active_tracks[track_id]["last_bbox"] = bbox

        # ── Track Lost Handling ──
        lost_tracks = []
        for track_id, track_info in list(self.active_tracks.items()):
            if track_id not in current_tracks:
                events.append({
                    "event_type": "TRACK_LOST",
                    "track_id": track_id,
                    "class_name": track_info.get("class_name", "object"),
                    "camera": self.source_name,
                    "timestamp": now,
                })
                lost_tracks.append(track_id)

        for track_id in lost_tracks:
            self.active_tracks.pop(track_id, None)

        return events


event_engine = EventEngine()