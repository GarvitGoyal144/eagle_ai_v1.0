import gc
import sys
import time
import cv2
import uuid
from typing import Dict, Any, List

from app.services.detection_service import detection_service
from app.services.event_engine import event_engine
from app.services.event_service import event_service

# Global in-memory progress tracker accessible by API route
progress_store: Dict[str, Dict[str, Any]] = {}

# Mini-batch size: process this many frames at a time to stay within 512MB RAM on Render
# Each 1080p frame ≈ 6MB; 8 frames ≈ 48MB — safe alongside YOLO's ~200MB
MINI_BATCH_SIZE = 8


def log(msg: str):
    """Flush-safe logging that always appears in Render logs."""
    print(msg, flush=True)
    sys.stdout.flush()


class VideoProcessor:
    """
    Memory-Safe YOLO Surveillance Video Processor.
    Processes frames in mini-batches to stay within Render's 512MB RAM limit.
    Provides live progress updates throughout all phases.
    """

    def process(self, video_path: str, filename: str, session_id: str | None = None) -> Dict[str, Any]:
        log(f"🎬 Starting YOLO analysis for {filename}...")
        start_time = time.time()
        if not session_id:
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
        duration = round(total_frames / fps, 1) if fps > 0 else 0
        log(f"📊 Video: {total_frames} frames @ {fps:.0f}fps = {duration}s")

        # 1.5 FPS sampling: for a 74s video, processes ~111 frames
        target_fps = 1.5
        skip_frames = max(1, int(fps / target_fps))
        sampled_indices = list(range(0, max(1, total_frames), skip_frames))
        total_samples = len(sampled_indices)
        log(f"⚙️  Analyzing {total_samples} key frames in mini-batches of {MINI_BATCH_SIZE}")

        event_engine.set_source(filename)

        progress_store[filename] = {
            "status": "processing",
            "progress": 5,
            "current_frame": 0,
            "total_frames": total_frames,
            "detections": 0,
            "unique_tracks": 0,
            "elapsed_sec": 0.0,
            "step": f"Starting YOLO analysis on {total_samples} frames...",
        }

        total_detections = 0
        unique_tracks = set()
        class_counts: Dict[str, int] = {}
        all_events: List[Dict[str, Any]] = []
        sample_idx = 0

        # ── Mini-batch processing: read → detect → process → free RAM ──
        for batch_start in range(0, total_samples, MINI_BATCH_SIZE):
            batch_indices = sampled_indices[batch_start: batch_start + MINI_BATCH_SIZE]

            # Read this mini-batch of frames
            batch_frames = []
            for frame_idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    batch_frames.append((frame_idx, frame))

            if not batch_frames:
                sample_idx += len(batch_indices)
                continue

            # YOLO batch detect on this mini-batch only
            try:
                frames_only = [f for _, f in batch_frames]
                batch_detections = detection_service.detect_batch(frames_only)
                del frames_only
            except Exception as exc:
                log(f"⚠️ Batch detection error: {exc}")
                for _, frame in batch_frames:
                    del frame
                del batch_frames
                gc.collect()
                sample_idx += len(batch_indices)
                continue

            # Process detections + generate events
            for (frame_idx, frame), detections in zip(batch_frames, batch_detections):
                sample_idx += 1
                timestamp_sec = round(frame_idx / fps, 2)

                total_detections += len(detections)
                for d in detections:
                    unique_tracks.add(d["track_id"])
                    cls_name = d["class_name"]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                try:
                    events = event_engine.process(detections)
                except Exception:
                    events = []

                for event in events:
                    event["frame_number"] = frame_idx
                    event["timestamp_sec"] = timestamp_sec
                    event["video_filename"] = filename
                    event["session_id"] = session_id
                    all_events.append(event)

                del frame

            # Free mini-batch frames immediately after processing
            del batch_frames
            del batch_detections
            gc.collect()

            # Live progress update after each mini-batch
            elapsed = round(time.time() - start_time, 1)
            pct = min(95, max(10, int((sample_idx / total_samples) * 100)))
            progress_store[filename] = {
                "status": "processing",
                "progress": pct,
                "current_frame": sampled_indices[min(batch_start + MINI_BATCH_SIZE - 1, total_samples - 1)],
                "total_frames": total_frames,
                "detections": total_detections,
                "unique_tracks": len(unique_tracks),
                "elapsed_sec": elapsed,
                "step": f"Batch {batch_start // MINI_BATCH_SIZE + 1}/{(total_samples + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE} • {len(unique_tracks)} objects found",
            }
            log(f"  ✓ Mini-batch done: {sample_idx}/{total_samples} frames | {pct}% | {elapsed}s elapsed")

        cap.release()
        gc.collect()

        # Save all events in one bulk operation
        if all_events:
            event_service.save_events_bulk(all_events)

        proc_time = round(time.time() - start_time, 2)
        log(f"⚡ Analysis complete in {proc_time}s — {len(all_events)} events, {len(unique_tracks)} unique objects")

        insights = {
            "filename": filename,
            "duration_seconds": duration,
            "total_frames_sampled": total_samples,
            "total_detections": total_detections,
            "unique_tracks": len(unique_tracks),
            "class_counts": class_counts,
            "events_saved": len(all_events),
            "scene_snapshots_saved": 0,
            "processing_time_seconds": proc_time,
            "session_id": session_id,
        }

        # Mark 100% complete — insights stored in progress_store for frontend polling
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
