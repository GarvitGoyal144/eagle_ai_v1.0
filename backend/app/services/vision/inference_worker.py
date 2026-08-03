import threading
import time

from app.config.settings import settings


class InferenceWorker:
    """
    Runs YOLO + tracking on a background thread so the MJPEG
    stream loop stays responsive.

    Also handles throttled CLIP encoding:
    - Scene snapshots every CLIP_SCENE_INTERVAL seconds
    - Crop embeddings on new-track events (PERSON_ENTERED, OBJECT_DETECTED)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self._detections = []
        self._running = False
        self._thread = None

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            self._latest_frame = None
            self._detections = []

    def submit_frame(self, frame):
        if not self._running:
            return
        with self._lock:
            self._latest_frame = frame

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def _loop(self):
        from app.services.detection_service import detection_service
        from app.services.event_engine import event_engine
        from app.services.event_service import event_service
        from app.services.embeddings.embedding_engine import embedding_engine

        frame_interval = 1.0 / max(settings.INFERENCE_FPS, 1)
        last_scene_encode = 0.0

        while True:
            with self._lock:
                if not self._running:
                    break
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            started = time.perf_counter()

            try:
                # ── YOLO detection + ByteTrack ──
                detections = detection_service.track(frame)
                events = event_engine.process(detections)

                # ── Vision Encoder: Scene snapshot & crop embeddings (throttled) ──
                if not settings.DISABLE_CLIP:
                    now = time.time()
                    if now - last_scene_encode >= settings.CLIP_SCENE_INTERVAL:
                        self._encode_scene(frame, now, event_service, embedding_engine)
                        last_scene_encode = now

                    if events:
                        self._encode_crops(frame, events, embedding_engine)

                if events:
                    threading.Thread(
                        target=event_service.save_events,
                        args=(events,),
                        daemon=True,
                    ).start()

                with self._lock:
                    self._detections = detections
            except Exception as exc:
                print(f"Inference error: {exc}")

            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, frame_interval - elapsed))

    @staticmethod
    def _encode_scene(frame, timestamp, event_service, embedding_engine):
        """Encode the full frame with CLIP and store as a scene snapshot."""
        try:
            embedding = embedding_engine.encode_frame(frame)
            event_service.save_scene_embedding(embedding, timestamp)
        except Exception as exc:
            print(f"Scene encoding error: {exc}")

    @staticmethod
    def _encode_crops(frame, events, embedding_engine):
        """Attach CLIP crop embeddings to new-track events."""
        for event in events:
            if event.get("event_type") not in ("PERSON_ENTERED", "OBJECT_DETECTED"):
                continue
            bbox = event.get("bbox")
            if bbox is None:
                continue
            try:
                crop_emb = embedding_engine.encode_crop(frame, bbox)
                if crop_emb is not None:
                    event["embedding"] = crop_emb.tolist()
            except Exception as exc:
                print(f"Crop encoding error (track #{event.get('track_id')}): {exc}")


inference_worker = InferenceWorker()
