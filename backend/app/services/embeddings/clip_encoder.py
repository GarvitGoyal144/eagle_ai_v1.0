import os
import cv2
import numpy as np

# Suppress HF unauthenticated token notice in logs
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app.config.settings import settings
from app.services.embeddings.base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._device = "cpu"

        if settings.DISABLE_CLIP:
            print("✅ CLIP disabled — running in keyword-only mode (memory-safe for Render free tier).")
            return

        # Only import heavy libraries when CLIP is actually needed
        # This keeps ~150MB of torch + open_clip off the heap when CLIP is disabled
        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() and settings.DEVICE.lower() != "cpu" else "cpu"
            torch.set_num_threads(1)  # Cap CPU thread pool — saves ~30MB on cloud

            model_name = settings.CLIP_MODEL or "ViT-B-32-quickgelu"
            print(f"Loading CLIP ({model_name}) on {self._device.upper()}...")

            pretrained = "openai"
            if "MobileCLIP2" in model_name:
                pretrained = "dfndr2b"
            elif "MobileCLIP" in model_name:
                pretrained = "datacompdr"

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.to(self._device)
            self.model.eval()
            print(f"CLIP ({model_name}) Loaded ✅")

        except Exception as exc:
            print(f"⚠️ Warning: Could not load CLIP model: {exc}")
            print("Falling back to keyword-only metadata mode.")
            self.model = None

    def encode_image(self, frame):
        if self.model is None:
            return np.zeros(512, dtype=np.float32)

        try:
            import torch
            from PIL import Image
            with torch.no_grad():
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image = self.preprocess(image).unsqueeze(0).to(self._device)
                features = self.model.encode_image(image)
                features /= features.norm(dim=-1, keepdim=True)
                return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode image error: {exc}")
            return np.zeros(512, dtype=np.float32)

    def encode_text(self, text):
        if self.model is None:
            return np.zeros(512, dtype=np.float32)

        try:
            import torch
            with torch.no_grad():
                tokens = self.tokenizer([text]).to(self._device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode text error: {exc}")
            return np.zeros(512, dtype=np.float32)