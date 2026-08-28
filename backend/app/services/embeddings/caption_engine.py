import json
import hashlib
from pathlib import Path
import numpy as np

from app.config.settings import settings
from app.services.embeddings.clip_encoder import CLIPEncoder

# Disk cache directory for pre-encoded embedding matrices
_CACHE_DIR = Path("data/embedding_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _content_hash(items: list[str]) -> str:
    """MD5 hash of the captions list — used to invalidate cache if captions change."""
    return hashlib.md5("|".join(items).encode()).hexdigest()[:12]


def _load_cache(name: str, content_hash: str) -> np.ndarray | None:
    path = _CACHE_DIR / f"{name}_{content_hash}.npy"
    if path.exists():
        try:
            vec = np.load(str(path))
            print(f"⚡ Loaded cached {name} embeddings ({len(vec)} vectors) from disk", flush=True)
            return vec
        except Exception:
            pass
    return None


def _save_cache(name: str, content_hash: str, matrix: np.ndarray):
    try:
        path = _CACHE_DIR / f"{name}_{content_hash}.npy"
        np.save(str(path), matrix)
        print(f"💾 Cached {name} embeddings ({len(matrix)} vectors) to disk", flush=True)
    except Exception as e:
        print(f"⚠️ Cache save warning: {e}", flush=True)


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
    "truck or large vehicle",
]

# Rich surveillance & incident categories for zero-shot scene classification
DEFAULT_INCIDENT_CAPTIONS = [
    # Normal operations
    ("normal traffic flow with cars and vehicles moving on the road", "normal: traffic"),
    ("vehicles parked or driving normally along the street", "normal: vehicles"),
    ("a person walking normally on the sidewalk or pedestrian area", "normal: pedestrian"),
    ("multiple people walking peacefully in a public area", "normal: pedestrians"),
    # Traffic incidents & Accidents
    ("a car accident with damaged vehicles or a collision on the road", "incident: car accident"),
    ("two vehicles crashing or colliding with each other", "incident: vehicle collision"),
    ("a wrecked, damaged, or overturned car after an accident", "incident: crash damage"),
    ("vehicles stopped abruptly on the road due to an accident or hazard", "incident: traffic obstruction"),
    ("a vehicle driving off the road or into a barrier", "incident: reckless driving"),
    # Security & Pedestrian Incidents
    ("a person lying on the ground, fallen, or injured", "incident: fallen person"),
    ("people fighting, assaulting, or engaged in a physical altercation", "incident: physical altercation"),
    ("a crowd of people running or fleeing from an area", "incident: crowd panic"),
    ("smoke, fire, or flames coming from a vehicle or structure", "incident: fire/smoke"),
]


class CaptionEngine:
    """
    Zero-Shot Visual Caption & Attribute Engine.

    1. Loads dataset captions from `merged_captions.json` + incident templates
    2. Pre-encodes text vectors into a tensor matrix at startup (0ms runtime math)
    3. Performs fast dot-product classification on image crops and scene frames
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
        print("🧠 Initializing Visual Caption & Attribute Engine...", flush=True)

        # ── 1. Load incident captions ──
        for caption, category in DEFAULT_INCIDENT_CAPTIONS:
            self._captions.append(caption)
            self._caption_categories.append(category)

        # ── 2. Load dataset captions from merged_captions.json if present ──
        dataset_path = Path("data/merged_captions.json")
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parents[4] / "data" / "merged_captions.json"

        if dataset_path.exists():
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                descriptions = data.get("descriptions", [])
                seen = set(self._captions)

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

                        if len(self._captions) >= 300:
                            break

                print(f"Loaded {len(self._captions)} curated captions (including traffic incident templates)", flush=True)
            except Exception as exc:
                print(f"Dataset loading note: {exc}", flush=True)

        # ── 3. Pre-encode caption text vectors (with disk cache) ──
        cap_hash = _content_hash(self._captions)
        cached_captions = _load_cache("captions", cap_hash)
        if cached_captions is not None:
            self._caption_matrix = cached_captions
        else:
            print(f"Computing {len(self._captions)} caption embeddings via HF API...", flush=True)
            caption_vectors = [encoder.encode_text(c) for c in self._captions]
            self._caption_matrix = np.array(caption_vectors)
            _save_cache("captions", cap_hash, self._caption_matrix)

        # ── 4. Pre-encode visual attribute vectors (with disk cache) ──
        attr_hash = _content_hash(self._attribute_labels)
        cached_attrs = _load_cache("attributes", attr_hash)
        if cached_attrs is not None:
            self._attribute_matrix = cached_attrs
        else:
            print(f"Computing {len(self._attribute_labels)} attribute embeddings via HF API...", flush=True)
            attribute_vectors = [encoder.encode_text(a) for a in self._attribute_labels]
            self._attribute_matrix = np.array(attribute_vectors)
            _save_cache("attributes", attr_hash, self._attribute_matrix)

        self._loaded = True
        print("✅ Visual Caption & Attribute Engine pre-encoded and ready (0ms runtime math)", flush=True)

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

        for idx in top_indices[:2]:
            if scores[idx] > 0.22:
                label = self._attribute_labels[idx]
                clean_label = label.replace("person ", "").replace("car or vehicle", "vehicle")
                matches.append(clean_label)

        return matches

    def classify_scene(self, scene_embedding: np.ndarray) -> dict:
        """
        Match full video frame snapshot to surveillance & incident captions.
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
