from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    """
    Central configuration class for Eagle AI.

    Every configurable value in the backend should
    come from this class instead of being hardcoded.
    """

    PROJECT_NAME = os.getenv("PROJECT_NAME", "Eagle AI")
    PROJECT_VERSION = os.getenv("PROJECT_VERSION", "1.0.0")

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 8000))

    MONGO_URI = os.getenv("MONGO_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER"))
    OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER"))
    CLIP_FOLDER = Path(os.getenv("CLIP_FOLDER"))

    YOLO_MODEL = os.getenv("YOLO_MODEL")
    CLIP_MODEL = os.getenv("CLIP_MODEL")

    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    LLM_MODEL = os.getenv("LLM_MODEL")

    DEVICE = os.getenv("DEVICE", "cpu")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()