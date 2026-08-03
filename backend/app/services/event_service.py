from datetime import datetime, timezone
import uuid

from pymongo import TEXT

from app.database.mongodb import mongodb


class EventService:

    def init_indexes(self):
        """Ensure text indexes exist for hybrid search."""
        if mongodb.database is None:
            return
        try:
            mongodb.database.events.create_index(
                [("class_name", TEXT), ("event_type", TEXT), ("camera", TEXT)],
                name="text_search_index",
                background=True,
            )
        except Exception as exc:
            print(f"Index creation note: {exc}")

    def save_events(self, events):
        """Save detection/tracking events to MongoDB."""
        if mongodb.database is None:
            return

        for event in events:
            event["event_id"] = str(uuid.uuid4())
            mongodb.database.events.insert_one(event)
            print(f"✅ {event['event_type']}  Track #{event['track_id']}")

    def save_scene_embedding(self, embedding, timestamp: float):
        """
        Save a scene embedding snapshot to MongoDB.
        These are full-frame embeddings captured every N seconds.
        """
        if mongodb.database is None:
            return

        doc = {
            "snapshot_id": str(uuid.uuid4()),
            "embedding": embedding.tolist(),
            "camera": "webcam",
            "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
        }

        mongodb.database.scene_embeddings.insert_one(doc)


event_service = EventService()