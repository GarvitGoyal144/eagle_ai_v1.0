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
        model_name = settings.CLIP_MODEL or "ViT-B-32"

        print(f"Loading CLIP ({model_name}) on {self.device.upper()}...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"CLIP ({model_name}) Loaded ✅")

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