import threading
import numpy as np

from app.config.settings import settings
from app.services.embeddings.caption_engine import caption_engine


class EmbeddingEngine:
    """
    Central interface for CLIP vision/text encoding & visual attribute classification.
    Lazy loads model and caption engine at startup.
    """

    def __init__(self):
        self._encoder = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        """Load the encoder and caption engine on first use (thread-safe)."""
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return  # double-check after acquiring lock
            from app.services.embeddings.encoder_factory import EncoderFactory
            self._encoder = EncoderFactory.create()
            caption_engine.initialize(self._encoder)

    def encode_frame(self, frame) -> np.ndarray:
        """Encode a full video frame into a CLIP embedding vector."""
        self._ensure_loaded()
        return self._encoder.encode_image(frame)

    def encode_crop(self, frame, bbox) -> np.ndarray | None:
        """
        Crop a bounding box region from the frame and encode it.
        Returns None if crop is invalid or too small.
        """
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1

        if w < settings.CLIP_CROP_MIN_SIZE or h < settings.CLIP_CROP_MIN_SIZE:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        self._ensure_loaded()
        return self._encoder.encode_image(crop)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a text query into a CLIP embedding vector."""
        self._ensure_loaded()
        return self._encoder.encode_text(text)

    def classify_scene_caption(self, frame) -> dict:
        """
        Generate zero-shot dataset caption for a full video frame.
        0.0ms matrix overhead.
        """
        self._ensure_loaded()
        emb = self.encode_frame(frame)
        res = caption_engine.classify_scene(emb)
        return {"embedding": emb, **res}

    def classify_crop_attributes(self, frame, bbox) -> tuple[np.ndarray | None, list[str]]:
        """
        Generate zero-shot visual attributes (e.g. blue clothing, backpack) for an object crop.
        0.0ms matrix overhead.
        """
        self._ensure_loaded()
        emb = self.encode_crop(frame, bbox)
        if emb is None:
            return None, []
        attrs = caption_engine.classify_crop(emb)
        return emb, attrs


embedding_engine = EmbeddingEngine()