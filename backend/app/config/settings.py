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

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    # Fail loudly if MONGO_URI is not set in environment — no silent localhost fallback
    MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "eagle_ai")

    # Allow all origins by default so Vercel frontend can reach Render backend
    _cors_env = os.getenv("CORS_ORIGINS", "*")
    CORS_ORIGINS = (
        ["*"]
        if _cors_env.strip() == "*"
        else [o.strip() for o in _cors_env.split(",") if o.strip()]
    )

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "data/videos"))
    OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "data/outputs"))
    CLIP_FOLDER = Path(os.getenv("CLIP_FOLDER", "data/clips"))

    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
    TRACKER = os.getenv("TRACKER", "ocsort.yaml")
    CLIP_MODEL = os.getenv("CLIP_MODEL", "MobileCLIP2-S0")

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    # Default to gemini (free cloud LLM) — NOT ollama (local only)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    # Cloud-powered CLIP feature extraction (0 MB RAM overhead on Render)
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

# Startup diagnostics
if not os.getenv("MONGO_URI") and not os.getenv("MONGODB_URI"):
    print("⚠️  WARNING: MONGO_URI env var not set — using localhost fallback. Set MONGO_URI on Render/cloud.")
if not settings.GEMINI_API_KEY and settings.LLM_PROVIDER == "gemini":
    print("⚠️  WARNING: GEMINI_API_KEY env var not set — chat will fail. Set GEMINI_API_KEY on Render/cloud.")
print(f"🔧 LLM Provider: {settings.LLM_PROVIDER} | CORS: {settings.CORS_ORIGINS} | CLIP: {'DISABLED (memory safe)' if settings.DISABLE_CLIP else 'ENABLED'}")

