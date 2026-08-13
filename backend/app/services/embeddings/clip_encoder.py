import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from app.config.settings import settings
from app.services.embeddings.base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.DEVICE.lower() != "cpu" else "cpu"
        # Constrain CPU thread count on cloud to save RAM
        if self.device == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        model_name = settings.CLIP_MODEL or "ViT-B-32-quickgelu"
        self.model = None
        self.tokenizer = None
        self.preprocess = None

        if settings.DISABLE_CLIP:
            print("⚠️ CLIP disabled via DISABLE_CLIP setting.")
            return

        print(f"Loading CLIP ({model_name}) on {self.device.upper()}...")

        try:
            # Handle open_clip pretrained parameters gracefully
            pretrained = "openai"
            if "MobileCLIP" in model_name:
                pretrained = "dfn1b"

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

            self.model.to(self.device)
            self.model.eval()
            print(f"CLIP ({model_name}) Loaded ✅")
        except Exception as exc:
            print(f"⚠️ Warning: Could not load CLIP model ({model_name}): {exc}")
            print("Falling back to text/keyword metadata mode.")
            self.model = None

    @torch.no_grad()
    def encode_image(self, frame):
        if self.model is None:
            return np.zeros(512, dtype=np.float32)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image = self.preprocess(image).unsqueeze(0).to(self.device)

            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)

            return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode image error: {exc}")
            return np.zeros(512, dtype=np.float32)

    @torch.no_grad()
    def encode_text(self, text):
        if self.model is None:
            return np.zeros(512, dtype=np.float32)

        try:
            tokens = self.tokenizer([text]).to(self.device)

            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

            return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode text error: {exc}")
            return np.zeros(512, dtype=np.float32)