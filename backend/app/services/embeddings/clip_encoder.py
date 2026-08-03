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

        model_name = settings.CLIP_MODEL or "ViT-B-16-SigLIP-2"
        pretrained = "webli" if "siglip" in model_name.lower() else "openai"

        print(f"Loading Vision Encoder ({model_name}) on {self.device.upper()}...")

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            print(f"⚠️ Failed to load {model_name} ({exc}), falling back to ViT-B-32/openai...")
            model_name = "ViT-B-32"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.model.to(self.device)
        self.model.eval()

        print(f"Vision Encoder ({model_name}) Loaded ✅")

    @torch.no_grad()
    def encode_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        features = self.model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)

        features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()