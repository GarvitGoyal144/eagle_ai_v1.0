import gc
import time
import cv2
import uuid
from typing import Dict, Any

from app.config.settings import settings
from app.services.detection_service import detection_service
from app.services.event_engine import event_engine
from app.services.event_service import event_service
from app.services.embeddings.embedding_engine import embedding_engine

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
        Memory-safe: explicitly deletes each frame after use and runs gc.collect().
        """
        print(f"🎬 Starting processing for {filename} at {video_path}")
        start_time = time.time()
        session_id = str(uuid.uuid4())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video {video_path}")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Hard cap: process at max 2 FPS equivalent on Render free tier
        # This keeps video processing memory predictable and safe
        target_fps = 2.0 if settings.DISABLE_CLIP else 5.0
        skip_frames = max(1, int(fps / target_fps))

        # Stats
        total_detections = 0
        unique_tracks = set()
        class_counts = {}
        events_saved = 0
        scene_snapshots_saved = 0
        frames_sampled = 0

        last_scene_encode = -settings.CLIP_SCENE_INTERVAL
        gc_interval = 10  # Log progress every 10 processed frames so we can see it's moving

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx % skip_frames != 0:
                # Explicitly release skipped frames immediately
                del frame
                continue

            frames_sampled += 1
            timestamp_sec = round(frame_idx / fps, 2)

            # 1. YOLO Detection
            try:
                detections = detection_service.track(frame)
            except Exception as exc:
                print(f"Detection error at frame {frame_idx}: {exc}")
                del frame
                continue

            total_detections += len(detections)
            for d in detections:
                unique_tracks.add(d["track_id"])
                cls_name = d["class_name"]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # 2. Event Generation
            events = event_engine.process(detections, source_name=filename)

            # Always tag events with video metadata for on-demand visual retrieval
            for event in events:
                event["frame_number"] = frame_idx
                event["timestamp_sec"] = timestamp_sec
                event["video_filename"] = filename
                event["session_id"] = session_id

            # 3. Vision Encoding (only when CLIP is enabled)
            if not settings.DISABLE_CLIP:
                # Scene snapshot every N seconds
                if timestamp_sec - last_scene_encode >= settings.CLIP_SCENE_INTERVAL:
                    try:
                        res = embedding_engine.classify_scene_caption(frame)
                        event_service.save_scene_embedding(
                            embedding=res["embedding"],
                            timestamp=time.time(),
                            caption=res.get("caption", ""),
                            category=res.get("category", "normal"),
                            camera=filename,
                            snapshot_id=f"scene_{session_id}_{frame_idx}"
                        )
                        scene_snapshots_saved += 1
                        last_scene_encode = timestamp_sec
                    except Exception as e:
                        print(f"Scene encode error: {e}")

                # Event embeddings
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
                                print(f"Event encoding error: {exc}")

            # 4. Save Events
            if events:
                event_service.save_events(events)
                events_saved += len(events)

            # ✅ KEY: explicitly release frame memory after each processed frame
            del frame

            # Force garbage collection periodically to prevent accumulation
            if frames_sampled % gc_interval == 0:
                gc.collect()
                print(f"  → Frame {frame_idx}/{total_frames} | {frames_sampled} processed | {events_saved} events saved")

        cap.release()

        # Final cleanup
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

        print(f"✅ Finished processing {filename} in {proc_time}s | {events_saved} events saved")
        return insights

video_processor = VideoProcessor()
