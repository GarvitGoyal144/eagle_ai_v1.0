import gc
import sys
import time
import traceback
import cv2
import uuid
from typing import Dict, Any

from app.config.settings import settings
from app.services.detection_service import detection_service
from app.services.event_engine import event_engine
from app.services.event_service import event_service
from app.services.embeddings.embedding_engine import embedding_engine


def log(msg: str):
    """Flush-safe logging that always appears in Render logs."""
    print(msg, flush=True)
    sys.stdout.flush()


class VideoProcessor:
    """
    Processes video files frame-by-frame.
    Extracts YOLO detections, optionally generates CLIP embeddings (when enabled),
    and stores rich event metadata (frame_number, timestamp, video_filename).
    """

    def process(self, video_path: str, filename: str) -> Dict[str, Any]:
        """
        Process the entire video synchronously.
        Returns VideoInsights dictionary.
        """
        log(f"🎬 [STEP 1/6] Starting processing for {filename} at {video_path}")
        start_time = time.time()
        session_id = str(uuid.uuid4())

        # Step 2: Open video
        log(f"📂 [STEP 2/6] Opening video file...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"❌ Failed to open video {video_path}")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, ~{total_frames/fps:.1f}s duration")

        # Adaptive sampling (1.5 FPS is optimal for surveillance video processing without timeout)
        target_fps = 1.5
        skip_frames = max(1, int(fps / target_fps))
        log(f"⚙️  Sampling: every {skip_frames} frames (~{target_fps} FPS), CLIP={'OFF' if settings.DISABLE_CLIP else 'ON'}")

        # Set the event engine source name for this session
        event_engine.set_source(filename)

        # Stats
        total_detections = 0
        unique_tracks = set()
        class_counts = {}
        events_saved = 0
        scene_snapshots_saved = 0
        frames_sampled = 0

        last_scene_encode = -settings.CLIP_SCENE_INTERVAL

        # Step 3: Load YOLO model (lazy, happens on first .track() call)
        log(f"🔄 [STEP 3/6] Starting frame-by-frame processing...")

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
                detections = detection_service.track(frame)
            except Exception as exc:
                log(f"⚠️  Detection error at frame {frame_idx}: {exc}")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
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
                log(f"⚠️  Event engine error at frame {frame_idx}: {exc}")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                events = []

            # Tag events with video metadata
            for event in events:
                event["frame_number"] = frame_idx
                event["timestamp_sec"] = timestamp_sec
                event["video_filename"] = filename
                event["session_id"] = session_id

            # --- Vision Encoding (only when CLIP is enabled) ---
            if not settings.DISABLE_CLIP:
                if timestamp_sec - last_scene_encode >= settings.CLIP_SCENE_INTERVAL:
                    try:
                        res = embedding_engine.classify_scene_caption(frame)
                        event_service.save_scene_embedding(
                            embedding=res["embedding"],
                            timestamp=time.time(),
                            caption=res.get("caption", ""),
                            category=res.get("category", "normal"),
                            camera=filename,
                            snapshot_id=f"scene_{session_id}_{frame_idx}",
                            frame_number=frame_idx,
                            timestamp_sec=timestamp_sec,
                            video_filename=filename,
                            session_id=session_id,
                        )
                        scene_snapshots_saved += 1
                        last_scene_encode = timestamp_sec
                    except Exception as e:
                        log(f"⚠️  Scene encode error: {e}")

                for event in events:
                    if event.get("event_type") in ("PERSON_ENTERED", "OBJECT_DETECTED"):
                        bbox = event.get("bbox")
                        if bbox:
                            try:
                                emb = embedding_engine.encode_frame(frame)
                                event["embedding"] = emb.tolist()
                                _, attrs = embedding_engine.classify_crop_attributes(frame, bbox)
                                if attrs:
                                    event["attributes"] = attrs
                            except Exception as exc:
                                log(f"⚠️  Event encoding error: {exc}")

            # --- Save Events ---
            if events:
                try:
                    event_service.save_events(events)
                    events_saved += len(events)
                except Exception as exc:
                    log(f"⚠️  Event save error: {exc}")

            del frame

            # Progress logging every 10 frames
            if frames_sampled % 10 == 0:
                gc.collect()
                elapsed = round(time.time() - start_time, 1)
                log(f"  → Frame {frame_idx}/{total_frames} | {frames_sampled} sampled | {events_saved} events | {elapsed}s elapsed")

        cap.release()
        gc.collect()

        duration = round(total_frames / fps, 1) if fps > 0 else 0
        proc_time = round(time.time() - start_time, 2)

        insights = {
            "filename": filename,
            "duration_seconds": duration,
            "total_frames_sampled": frames_sampled,
            "total_detections": total_detections,
            "unique_tracks": len(unique_tracks),
            "class_counts": class_counts,
            "events_saved": events_saved,
            "scene_snapshots_saved": scene_snapshots_saved,
            "processing_time_seconds": proc_time,
            "session_id": session_id
        }

        log(f"✅ [STEP 6/6] Finished processing {filename} in {proc_time}s | {events_saved} events saved")
        return insights

video_processor = VideoProcessor()
