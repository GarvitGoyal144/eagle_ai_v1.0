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


def log(msg: str):
    """Flush-safe logging that always appears in Render logs."""
    print(msg, flush=True)
    sys.stdout.flush()


class VideoProcessor:
    """
    Ultra-Fast YOLO Surveillance Video Processor.
    Processes video in 1-3 seconds using direct frame seeking and bulk DB operations.
    """

    def process(self, video_path: str, filename: str, session_id: str | None = None) -> Dict[str, Any]:
        log(f"🎬 Starting ultra-fast YOLO analysis for {filename}...")
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

        # 1.5 FPS sampling: for a 20s video, processes exactly ~30 frames
        target_fps = 1.5
        skip_frames = max(1, int(fps / target_fps))
        sampled_indices = list(range(0, max(1, total_frames), skip_frames))
        total_samples = len(sampled_indices)
        log(f"⚙️  Analyzing {total_samples} key frames (every {skip_frames} frames)")

        event_engine.set_source(filename)

        progress_store[filename] = {
            "status": "processing",
            "progress": 5,
            "current_frame": 0,
            "total_frames": total_frames,
            "detections": 0,
            "unique_tracks": 0,
            "elapsed_sec": 0.0,
            "step": f"Analyzing {total_samples} surveillance frames with YOLOv11...",
        }

        total_detections = 0
        unique_tracks = set()
        class_counts: Dict[str, int] = {}
        all_events: List[Dict[str, Any]] = []

        # ── Phase 1: Read all sampled frames from disk ──
        frames_data = []  # list of (frame_idx, frame)
        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_data.append((frame_idx, frame))

        cap.release()
        gc.collect()

        if not frames_data:
            log(f"⚠️ No frames could be read from {filename}")
            return {}

        log(f"📦 Read {len(frames_data)} frames — running single batch YOLO inference...")

        # ── Phase 2: Batch YOLO detection (one model call for all frames) ──
        frames_list = [f for _, f in frames_data]
        batch_detections = detection_service.detect_batch(frames_list)
        del frames_list
        gc.collect()

        log(f"✅ Batch YOLO complete — processing events...")

        # ── Phase 3: Process events and update progress ──
        for sample_idx, ((frame_idx, frame), detections) in enumerate(zip(frames_data, batch_detections), 1):
            timestamp_sec = round(frame_idx / fps, 2)

            total_detections += len(detections)
            for d in detections:
                unique_tracks.add(d["track_id"])
                cls_name = d["class_name"]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            try:
                events = event_engine.process(detections)
            except Exception as exc:
                events = []

            for event in events:
                event["frame_number"] = frame_idx
                event["timestamp_sec"] = timestamp_sec
                event["video_filename"] = filename
                event["session_id"] = session_id
                all_events.append(event)

            del frame

            # Update live progress
            elapsed = round(time.time() - start_time, 1)
            pct = min(98, max(5, int((sample_idx / total_samples) * 100)))
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

        del frames_data
        del batch_detections
        gc.collect()

        # Save all generated events in one single fast batch
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

        # Mark 100% complete
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
