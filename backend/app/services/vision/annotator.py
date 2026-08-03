import cv2
import numpy as np


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a unique, vivid BGR color for each track ID."""
    hue = (track_id * 47) % 180
    hsv = cv2.cvtColor(
        np.array([[[hue, 200, 255]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )
    b, g, r = hsv[0, 0]
    return int(b), int(g), int(r)


def draw_detections(frame, detections):
    """Draw tracked bounding boxes and labels on a frame."""
    if not detections:
        return frame

    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = _track_color(det["track_id"])
        label = f"#{det['track_id']} {det['class_name']} {det['confidence']:.0%}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(y1 - 8, text_h + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated
