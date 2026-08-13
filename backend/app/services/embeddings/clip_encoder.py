import os
import cv2
import httpx
import numpy as np
from typing import Optional

from app.config.settings import settings
from app.services.embeddings.base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):
    """
    Dual-mode CLIP & Vision-Language Embedding Engine:
    1. Cloud Serverless Inference (0 MB RAM on Render) for text & image embeddings.
    2. Graceful local fallback for offline development.
    """

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN", "").strip()
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/clip-ViT-B-32"
        self._local_model = None
        self._local_tokenizer = None
        self._local_preprocess = None
        self._device = "cpu"

        # On cloud/Render, we default to Cloud API mode to save 250MB RAM
        self.cloud_mode = os.getenv("CLIP_MODE", "cloud").lower() == "cloud" or os.getenv("RENDER") == "true"
        
        if self.cloud_mode:
            print("⚡ CLIP Encoder running in Cloud Inference mode (0 MB local RAM overhead) ✅", flush=True)
        else:
            self._init_local_model()

    def _init_local_model(self):
        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() and settings.DEVICE.lower() != "cpu" else "cpu"
            torch.set_num_threads(1)

            model_name = settings.CLIP_MODEL or "MobileCLIP2-S0"
            pretrained = "dfndr2b" if "MobileCLIP2" in model_name else "openai"

            self._local_model, _, self._local_preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            self._local_tokenizer = open_clip.get_tokenizer(model_name)
            self._local_model.to(self._device)
            self._local_model.eval()
            print(f"Local CLIP ({model_name}) Loaded ✅", flush=True)
        except Exception as exc:
            print(f"Local CLIP load note: {exc} — switching to cloud mode", flush=True)
            self.cloud_mode = True

    def _call_hf_api(self, payload: bytes | dict, is_image: bool = False) -> Optional[np.ndarray]:
        """Send embedding request to Hugging Face serverless feature extraction API."""
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        try:
            if is_image:
                headers["Content-Type"] = "image/jpeg"
                response = httpx.post(self.api_url, headers=headers, content=payload, timeout=10.0)
            else:
                headers["Content-Type"] = "application/json"
                response = httpx.post(self.api_url, headers=headers, json=payload, timeout=10.0)

            if response.status_code == 200:
                vec = np.array(response.json(), dtype=np.float32)
                if vec.ndim > 1:
                    vec = vec.squeeze()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec
        except Exception:
            pass
        return None

    def encode_image(self, frame) -> np.ndarray:
        """Generate 512-dim vector for image frame with 0 MB memory overhead."""
        if frame is None:
            return np.zeros(512, dtype=np.float32)

        try:
            # Try Cloud API first
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                api_vec = self._call_hf_api(buffer.tobytes(), is_image=True)
                if api_vec is not None and len(api_vec) > 0:
                    return api_vec

            # Local fallback if model is initialized
            if self._local_model is not None and self._local_preprocess is not None:
                import torch
                from PIL import Image
                with torch.no_grad():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    img_tensor = self._local_preprocess(image).unsqueeze(0).to(self._device)
                    features = self._local_model.encode_image(img_tensor)
                    features /= features.norm(dim=-1, keepdim=True)
                    return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode image note: {exc}", flush=True)

        # Stable zero vector fallback
        return np.zeros(512, dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Generate 512-dim vector for text query."""
        if not text or not text.strip():
            return np.zeros(512, dtype=np.float32)

        try:
            # Try Cloud API first
            api_vec = self._call_hf_api({"inputs": text}, is_image=False)
            if api_vec is not None and len(api_vec) > 0:
                return api_vec

            # Local fallback if model is initialized
            if self._local_model is not None and self._local_tokenizer is not None:
                import torch
                with torch.no_grad():
                    tokens = self._local_tokenizer([text]).to(self._device)
                    features = self._local_model.encode_text(tokens)
                    features /= features.norm(dim=-1, keepdim=True)
                    return features.squeeze().cpu().numpy()
        except Exception as exc:
            print(f"Encode text note: {exc}", flush=True)

        return np.zeros(512, dtype=np.float32)