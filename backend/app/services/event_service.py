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

    def save_events(self, events):
        """Save detection/tracking events to MongoDB."""
        if mongodb.database is None:
            return

        for event in events:
            try:
                event["event_id"] = str(uuid.uuid4())
                mongodb.database.events.insert_one(event)

                attrs_str = (
                    f" [{', '.join(event['attributes'])}]"
                    if event.get("attributes")
                    else ""
                )
                cls_name = event.get("class_name", "object")
                evt_type = event.get("event_type", "EVENT")
                trk_id = event.get("track_id", "?")
                print(
                    f"✅ {evt_type}  Track #{trk_id} ({cls_name}){attrs_str}",
                    flush=True
                )
            except Exception as exc:
                print(f"Note: Could not save event to database: {exc}", flush=True)

    def save_scene_embedding(
        self,
        embedding,
        timestamp: float,
        caption: str = "",
        category: str = "normal",
        camera: str = "webcam",
        snapshot_id: str = "",
    ):
        """
        Save a scene embedding snapshot with dataset caption to MongoDB.
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
            }

            mongodb.database.scene_embeddings.insert_one(doc)
            if caption:
                print(f"📸 Scene Snapshot ({category}): \"{caption[:60]}...\"", flush=True)
        except Exception as exc:
            print(f"Note: Could not save scene embedding: {exc}", flush=True)


event_service = EventService()