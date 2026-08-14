import gc
import sys
import time
import cv2
import uuid
from typing import Dict, Any

from app.services.detection_service import detection_service
from app.services.event_engine import event_engine
from app.services.event_service import event_service

# Global in-memory progress tracker accessible by API route
progress_store: Dict[str, Dict[str, Any]] = {}


def log(msg: str):
    """Flush-safe logging that always appears in Render logs."""
    print(msg, flush=True)
    sys.stdout.flush()


class VideoProcessor:
    """
    Phase 1: YOLO-only local object detection with live progress tracking.
    Fast — runs entirely on-device with zero network latency.
    """

    def process(self, video_path: str, filename: str) -> Dict[str, Any]:
        log(f"🎬 Starting YOLO detection phase for {filename}...")
        start_time = time.time()
        session_id = str(uuid.uuid4())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"❌ Failed to open video {video_path}")
            progress_store[filename] = {
                "status": "error",
                "progress": 0,
                "step": "Failed to open video file",
            }
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log(f"📊 Video: {total_frames} frames @ {fps:.0f}fps = {total_frames/fps:.1f}s")

        # 2 FPS sampling rate
        skip_frames = max(1, int(fps / 2.0))
        log(f"⚙️  Sampling every {skip_frames} frames (2 FPS)")

        event_engine.set_source(filename)

        progress_store[filename] = {
            "status": "processing",
            "progress": 0,
            "current_frame": 0,
            "total_frames": total_frames,
            "detections": 0,
            "unique_tracks": 0,
            "elapsed_sec": 0.0,
            "step": f"Initializing YOLO on {total_frames} frames...",
        }

        total_detections = 0
        unique_tracks = set()
        class_counts: Dict[str, int] = {}
        events_saved = 0
        frames_sampled = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx % skip_frames != 0:
                del frame
                continue

            frames_sampled += 1
            timestamp_sec = round(frame_idx / fps, 2)

            # --- YOLO Detection ---
            try:
                detections = detection_service.detect(frame)
            except Exception as exc:
                log(f"⚠️ Detection error at frame {frame_idx}: {exc}")
                del frame
                continue

            total_detections += len(detections)
            for d in detections:
                unique_tracks.add(d["track_id"])
                cls_name = d["class_name"]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # --- Event Generation ---
            try:
                events = event_engine.process(detections)
            except Exception as exc:
                log(f"⚠️ Event engine error at frame {frame_idx}: {exc}")
                events = []

            for event in events:
                event["frame_number"] = frame_idx
                event["timestamp_sec"] = timestamp_sec
                event["video_filename"] = filename
                event["session_id"] = session_id

            if events:
                try:
                    event_service.save_events(events)
                    events_saved += len(events)
                except Exception as exc:
                    log(f"⚠️ Event save error: {exc}")

            del frame

            # Update live progress state
            elapsed = round(time.time() - start_time, 1)
            pct = min(99, max(1, int((frame_idx / max(1, total_frames)) * 100)))
            progress_store[filename] = {
                "status": "processing",
                "progress": pct,
                "current_frame": frame_idx,
                "total_frames": total_frames,
                "detections": total_detections,
                "unique_tracks": len(unique_tracks),
                "elapsed_sec": elapsed,
                "step": f"Frame {frame_idx}/{total_frames} • {len(unique_tracks)} objects detected",
            }

            if frames_sampled % 20 == 0:
                gc.collect()
                log(f"  → Frame {frame_idx}/{total_frames} | {pct}% | {len(unique_tracks)} objects | {elapsed}s")

        cap.release()
        gc.collect()

        proc_time = round(time.time() - start_time, 2)
        log(f"✅ YOLO phase done in {proc_time}s — {events_saved} events, {len(unique_tracks)} unique objects")

        insights = {
            "filename": filename,
            "duration_seconds": round(total_frames / fps, 1) if fps > 0 else 0,
            "total_frames_sampled": frames_sampled,
            "total_detections": total_detections,
            "unique_tracks": len(unique_tracks),
            "class_counts": class_counts,
            "events_saved": events_saved,
            "scene_snapshots_saved": 0,
            "processing_time_seconds": proc_time,
            "session_id": session_id,
        }

        # Mark completed
        progress_store[filename] = {
            "status": "completed",
            "progress": 100,
            "current_frame": total_frames,
            "total_frames": total_frames,
            "detections": total_detections,
            "unique_tracks": len(unique_tracks),
            "elapsed_sec": proc_time,
            "step": "Surveillance analysis complete!",
            "insights": insights,
        }

        return insights


video_processor = VideoProcessor()
