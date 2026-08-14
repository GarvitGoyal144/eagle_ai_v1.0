from datetime import datetime, timezone
import uuid

from pymongo import TEXT

from app.database.mongodb import mongodb
from app.services.event_engine import event_engine


class EventService:

    def init_indexes(self):
        """Ensure text indexes exist for hybrid search."""
        if mongodb.database is None:
            return
        try:
            mongodb.database.events.create_index(
                [
                    ("class_name", TEXT),
                    ("event_type", TEXT),
                    ("camera", TEXT),
                    ("caption", TEXT),
                    ("attributes", TEXT),
                ],
                name="text_search_index",
                background=True,
            )
        except Exception as exc:
            print(f"Index creation note: {exc}", flush=True)

    def get_events(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Fetch recent events from MongoDB with pagination support."""
        if mongodb.database is None:
            return []
        try:
            return list(
                mongodb.database.events.find(
                    {},
                    {"_id": 0}
                )
                .sort("timestamp", -1)
                .skip(offset)
                .limit(limit)
            )
        except Exception as exc:
            print(f"Database query note in get_events: {exc}", flush=True)
            return []

    def save_events_bulk(self, events: list[dict]):
        """Save a batch of detection events in a single network round-trip."""
        if mongodb.database is None or not events:
            return

        for ev in events:
            if "event_id" not in ev:
                ev["event_id"] = str(uuid.uuid4())

        try:
            mongodb.database.events.insert_many(events, ordered=False)
            print(f"✅ Saved {len(events)} events to database in single batch", flush=True)
        except Exception as exc:
            print(f"Note: Could not bulk save events: {exc}", flush=True)

    def save_events(self, events):
        """Save detection/tracking events to MongoDB (uses bulk save for speed)."""
        self.save_events_bulk(events)

    def save_scene_embedding(
        self,
        embedding,
        timestamp: float,
        caption: str = "",
        category: str = "normal",
        camera: str = "webcam",
        snapshot_id: str = "",
        frame_number: int = 0,
        timestamp_sec: float = 0.0,
        video_filename: str = "",
        session_id: str = "",
    ):
        """
        Save a scene embedding snapshot with dataset caption & video metadata to MongoDB.
        """
        if mongodb.database is None:
            return

        try:
            doc = {
                "snapshot_id": snapshot_id or str(uuid.uuid4()),
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                "caption": caption,
                "category": category,
                "camera": camera,
                "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                "frame_number": frame_number,
                "timestamp_sec": timestamp_sec,
                "video_filename": video_filename,
                "session_id": session_id,
            }

            mongodb.database.scene_embeddings.insert_one(doc)
            if caption:
                print(f"📸 Scene Snapshot ({category}): \"{caption[:60]}...\"", flush=True)
        except Exception as exc:
            print(f"Note: Could not save scene embedding: {exc}", flush=True)


event_service = EventService()