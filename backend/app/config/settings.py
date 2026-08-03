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

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "eagle_ai")

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "data/videos"))
    OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "data/outputs"))
    CLIP_FOLDER = Path(os.getenv("CLIP_FOLDER", "data/clips"))

    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
    TRACKER = os.getenv("TRACKER", "botsort.yaml")
    CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-16-SigLIP-2")

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "groq"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DISABLE_CLIP = os.getenv("DISABLE_CLIP", "false").lower() == "true"

    DEVICE = os.getenv("DEVICE", "cpu")

    # Live stream performance
    INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "480"))
    INFERENCE_FPS = int(os.getenv("INFERENCE_FPS", "8"))
    DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.4"))
    STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
    STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "480"))
    STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "72"))

    # CLIP semantic pipeline
    CLIP_SCENE_INTERVAL = float(os.getenv("CLIP_SCENE_INTERVAL", "3.0"))
    CLIP_CROP_MIN_SIZE = int(os.getenv("CLIP_CROP_MIN_SIZE", "40"))

    # Semantic search
    SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    VISION_MODEL: str = "clip"


settings = Settings()

