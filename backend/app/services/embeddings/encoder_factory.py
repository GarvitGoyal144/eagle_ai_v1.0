from app.config.settings import settings

from app.services.embeddings.clip_encoder import CLIPEncoder


class EncoderFactory:

    @staticmethod
    def create():

        if settings.VISION_MODEL.lower() == "clip":
            return CLIPEncoder()

        raise ValueError(
            f"Unknown Vision Model: {settings.VISION_MODEL}"
        )