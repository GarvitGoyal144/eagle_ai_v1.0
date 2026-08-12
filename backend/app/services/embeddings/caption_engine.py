import json
from pathlib import Path
import numpy as np

from app.config.settings import settings
from app.services.embeddings.clip_encoder import CLIPEncoder


# Standard visual attributes for YOLO detection crops
DEFAULT_ATTRIBUTES = [
    # Clothing / Appearance
    "person wearing black clothing",
    "person wearing white clothing",
    "person wearing blue clothing",
    "person wearing red clothing",
    "person wearing dark jacket",
    # Carried Objects
    "person carrying a backpack",
    "person carrying a handbag or purse",
    "person holding a smartphone",
    "person holding a laptop",
    "person with empty hands",
    # Vehicles
    "black car or vehicle",
    "white car or vehicle",
    "red car or vehicle",
    "blue car or vehicle",
    "silver or grey car",
    "motorcycle or bicycle",
]


class CaptionEngine:
    """
    Zero-Shot Visual Caption & Attribute Engine.

    1. Loads dataset captions from `merged_captions.json`
    2. Combines them with visual attribute templates
    3. Pre-encodes text vectors into a tensor matrix at startup (0ms runtime math)
    4. Performs fast dot-product classification on image crops and scene frames
    """

    def __init__(self):
        self._encoder = None
        self._captions = []
        self._caption_categories = []
        self._caption_matrix = None
        self._attribute_labels = list(DEFAULT_ATTRIBUTES)
        self._attribute_matrix = None
        self._loaded = False

    def initialize(self, encoder: CLIPEncoder):
        """Pre-encode captions and attributes into memory."""
        if self._loaded:
            return

        self._encoder = encoder
        print("🧠 Initializing Visual Caption & Attribute Engine...")

        # ── 1. Load dataset captions from merged_captions.json ──
        dataset_path = Path("data/merged_captions.json")
        if not dataset_path.exists():
            dataset_path = settings.BASE_DIR.parent / "data" / "merged_captions.json"

        if dataset_path.exists():
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                descriptions = data.get("descriptions", [])
                seen = set()

                # Curate top representative descriptions across categories
                for item in descriptions:
                    key = list(item.keys())[0] if item else None
                    if not key:
                        continue

                    category = item[key].get("category", key)
                    desc = item[key].get("description", "")
                    clean_desc = desc.replace("anomalous: ", "").replace("normal: ", "").strip()

                    if clean_desc and clean_desc not in seen and len(clean_desc) < 120:
                        seen.add(clean_desc)
                        self._captions.append(clean_desc)
                        self._caption_categories.append(f"{key}: {category}")

                        if len(self._captions) >= 300:  # sample top 300 captions for fast matrix math
                            break

                print(f"Loaded {len(self._captions)} curated captions from merged_captions.json")
            except Exception as exc:
                print(f"Dataset loading note: {exc}")

        # Fallback captions if dataset missing or small
        if len(self._captions) < 10:
            self._captions = [
                "a person walking normally through an indoor room",
                "a person standing in a hallway or entrance",
                "multiple people moving through a public area",
                "an individual forcefully pushing or shoving another person",
                "a person carrying a heavy bag or object",
                "a vehicle driving or parked in a lot",
            ]
            self._caption_categories = ["normal", "normal", "normal", "anomalous: assault", "normal", "normal"]

        # ── 2. Pre-encode caption text vectors ──
        caption_vectors = [encoder.encode_text(c) for c in self._captions]
        self._caption_matrix = np.array(caption_vectors)  # shape (N, 512)

        # ── 3. Pre-encode visual attribute vectors ──
        attribute_vectors = [encoder.encode_text(a) for a in self._attribute_labels]
        self._attribute_matrix = np.array(attribute_vectors)  # shape (M, 512)

        self._loaded = True
        print("✅ Visual Caption & Attribute Engine pre-encoded and ready (0ms runtime math)")

    def classify_crop(self, crop_embedding: np.ndarray) -> list[str]:
        """
        Classify visual attributes of an object crop (e.g. blue clothing, backpack).
        Runs in 0.0ms via matrix multiplication.
        """
        if not self._loaded or self._attribute_matrix is None or crop_embedding is None:
            return []

        # Matrix dot product
        scores = np.dot(crop_embedding, self._attribute_matrix.T)

        # Select top attributes above confidence threshold
        top_indices = np.argsort(scores)[::-1]
        matches = []

        for idx in top_indices[:2]:  # top 2 attributes
            if scores[idx] > 0.22:
                label = self._attribute_labels[idx]
                # Clean up label format
                clean_label = label.replace("person ", "").replace("car or vehicle", "vehicle")
                matches.append(clean_label)

        return matches

    def classify_scene(self, scene_embedding: np.ndarray) -> dict:
        """
        Match full video frame snapshot to dataset captions in merged_captions.json.
        Runs in 0.0ms via matrix multiplication.
        """
        if not self._loaded or self._caption_matrix is None or scene_embedding is None:
            return {"caption": "Visual frame captured", "category": "normal", "score": 0.0}

        # Matrix dot product
        scores = np.dot(scene_embedding, self._caption_matrix.T)
        best_idx = int(np.argmax(scores))

        return {
            "caption": self._captions[best_idx],
            "category": self._caption_categories[best_idx],
            "score": round(float(scores[best_idx]), 4),
        }


caption_engine = CaptionEngine()
