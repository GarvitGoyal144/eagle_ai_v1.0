import cv2

from app.services.detection_service import detection_service
from app.services.event_service import event_service



class CameraService:

    def __init__(self):

        self.camera = cv2.VideoCapture(0)

    def generate_frames(self):

        while True:

            success, frame = self.camera.read()

            if not success:
                break

            frame, detections = detection_service.detect(frame)
            event_service.save_new_tracks(detections)
            # asyncio.create_task(event_service.save_detections(detections)
            
            
#)

            _, buffer = cv2.imencode(".jpg", frame)

            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


camera_service = CameraService()