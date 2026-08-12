import cv2
import threading

from app.config.settings import settings
from app.services.vision.inference_worker import inference_worker


class CameraManager:

    def __init__(self):
        self.camera = None
        self.lock = threading.Lock()
        self.running = False
        self._ai_enabled = True  # AI features on by default

    def start(self):
        with self.lock:
            if self.running:
                return True

            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.STREAM_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.STREAM_HEIGHT)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.running = True

            # Only start inference if AI is enabled
            if self._ai_enabled:
                inference_worker.start()

            return True

    def stop(self):
        with self.lock:
            if not self.running and self.camera is None:
                return
            self.running = False

        # Stop inference worker outside lock to avoid deadlocks
        inference_worker.stop()

        with self.lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as exc:
                    print(f"Note releasing camera: {exc}")
                self.camera = None

    @property
    def is_running(self):
        return self.running

    @property
    def ai_enabled(self):
        return self._ai_enabled

    def set_ai_enabled(self, enabled: bool):
        """Toggle AI features on/off. Safe to call while streaming."""
        with self.lock:
            if self._ai_enabled == enabled:
                return  # no change

            self._ai_enabled = enabled

            if self.running:
                if enabled:
                    # Turn AI back on — start inference worker
                    inference_worker.start()
                else:
                    # Turn AI off — stop inference to free resources
                    inference_worker.stop()

    def get_frame(self):
        if not self.running:
            return None

        success, frame = self.camera.read()

        if not success:
            return None

        return frame


camera_manager = CameraManager()
