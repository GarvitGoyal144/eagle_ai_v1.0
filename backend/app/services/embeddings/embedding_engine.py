import threading

import numpy as np

from app.config.settings import settings


class EmbeddingEngine:
    """
    Central interface for CLIP vision/text encoding.

    The underlying encoder loads lazily on first call so that
    the FastAPI server starts quickly without waiting for CLIP
    to download and warm up.
    """

    def __init__(self):
        self._encoder = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        """Load the encoder on first use (thread-safe)."""
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return  # double-check after acquiring lock
            from app.services.embeddings.encoder_factory import EncoderFactory
            self._encoder = EncoderFactory.create()

    def encode_frame(self, frame) -> np.ndarray:
        """Encode a full video frame into a CLIP embedding vector."""
        self._ensure_loaded()
        return self._encoder.encode_image(frame)

    def encode_crop(self, frame, bbox) -> np.ndarray | None:
        """
        Crop a bounding box region from the frame and encode it.

        Returns None if the crop is too small or invalid.
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


embedding_engine = EmbeddingEngine()