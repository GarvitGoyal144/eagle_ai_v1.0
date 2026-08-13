from datetime import datetime, timezone


def _compute_iou(box1, box2) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


class EventEngine:
    """
    Converts raw detections into surveillance events:
    - Track appearances (PERSON_ENTERED, OBJECT_DETECTED)
    - Track disappearances (TRACK_LOST)
    - Vehicle collision / incident detection via spatial intersection heuristics
    """

    def __init__(self):
        self.active_tracks = {}
        self.source_name = "webcam"
        self.reported_collisions = set()

    def set_source(self, source: str):
        """Set the camera/video source name for event labeling and reset session state."""
        self.source_name = source
        self.active_tracks.clear()
        self.reported_collisions.clear()

    def process(self, detections):
        events = []
        current_tracks = set()
        now = datetime.now(timezone.utc)

        vehicle_classes = {"car", "truck", "bus", "motorcycle"}
        vehicles_in_frame = []

        for detection in detections:
            track_id = detection["track_id"]
            current_tracks.add(track_id)
            cls_name = detection.get("class_name", "object")
            bbox = detection.get("bbox", [])

            if cls_name in vehicle_classes and bbox:
                vehicles_in_frame.append((track_id, cls_name, bbox))

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

        # ── Collision / Overlap Heuristic ──
        # Check if any two vehicles have intersecting bounding boxes (potential crash/impact)
        for i in range(len(vehicles_in_frame)):
            for j in range(i + 1, len(vehicles_in_frame)):
                id1, cls1, box1 = vehicles_in_frame[i]
                id2, cls2, box2 = vehicles_in_frame[j]

                pair_key = tuple(sorted([id1, id2]))
                if pair_key in self.reported_collisions:
                    continue

                iou = _compute_iou(box1, box2)
                # If vehicle boxes overlap significantly (IoU > 0.15), trigger collision alert
                if iou > 0.15:
                    self.reported_collisions.add(pair_key)
                    events.append({
                        "event_type": "VEHICLE_COLLISION",
                        "track_id": id1,
                        "class_name": f"{cls1} and {cls2} collision",
                        "confidence": round(float(iou), 2),
                        "bbox": box1,
                        "camera": self.source_name,
                        "timestamp": now,
                        "attributes": [f"collision with Track #{id2}", "traffic incident"],
                    })

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