import cv2


class CameraService:

    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def generate_frames(self):

        while True:

            success, frame = self.camera.read()

            if not success:
                break

            _, buffer = cv2.imencode(".jpg", frame)

            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


camera_service = CameraService()