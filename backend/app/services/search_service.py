import re
import numpy as np

from app.config.settings import settings
from app.database.mongodb import mongodb
from app.services.embeddings.embedding_engine import embedding_engine


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class SearchService:
    """
    Hybrid Search Engine combining:
    1. Keyword/Text matching (MongoDB text & regex index)
    2. Semantic vector search (SigLIP / CLIP embeddings)

    Works instantly on fresh events even before embeddings are generated,
    and gains semantic depth as embeddings accumulate.
    """

    def keyword_search(self, query: str, limit: int = 50) -> list[dict]:
        """Keyword / regex matching over event fields."""
        if mongodb.database is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        results = []

        # Try MongoDB text index search first
        try:
            matched = list(
                mongodb.database.events.find(
                    {"$text": {"$search": clean_query}},
                    {"_id": 0, "embedding": 0, "score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )
            for doc in matched:
                results.append({
                    "type": "event",
                    "event_id": doc.get("event_id"),
                    "event_type": doc.get("event_type"),
                    "track_id": doc.get("track_id"),
                    "class_name": doc.get("class_name"),
                    "confidence": doc.get("confidence"),
                    "camera": doc.get("camera", "webcam"),
                    "timestamp": doc.get("timestamp"),
                    "score": round(min(1.0, float(doc.get("score", 0.5)) / 2.0), 4),
                    "search_type": "keyword",
                })
        except Exception:
            pass

        # Fallback to regex matching on class_name / event_type if text search yielded few results
        if len(results) < 3:
            pattern = re.compile(clean_query, re.IGNORECASE)
            regex_matched = list(
                mongodb.database.events.find(
                    {
                        "$or": [
                            {"class_name": pattern},
                            {"event_type": pattern},
                        ]
                    },
                    {"_id": 0, "embedding": 0},
                )
                .sort("timestamp", -1)
                .limit(limit)
            )

            existing_ids = {r.get("event_id") for r in results}
            for doc in regex_matched:
                if doc.get("event_id") not in existing_ids:
                    results.append({
                        "type": "event",
                        "event_id": doc.get("event_id"),
                        "event_type": doc.get("event_type"),
                        "track_id": doc.get("track_id"),
                        "class_name": doc.get("class_name"),
                        "confidence": doc.get("confidence"),
                        "camera": doc.get("camera", "webcam"),
                        "timestamp": doc.get("timestamp"),
                        "score": 0.8,
                        "search_type": "keyword",
                    })

        return results

    def semantic_search(self, query: str, top_k: int = 50) -> list[dict]:
        """Semantic vector search over scene & crop embeddings."""
        if mongodb.database is None or settings.DISABLE_CLIP:
            return []

        try:
            query_embedding = embedding_engine.encode_query(query)
        except Exception as exc:
            print(f"Embedding query error: {exc}")
            return []

        results = []

        # ── Search scene embeddings ──
        scenes = list(
            mongodb.database.scene_embeddings.find(
                {},
                {"_id": 0, "snapshot_id": 1, "embedding": 1, "camera": 1, "timestamp": 1},
            )
            .sort("timestamp", -1)
            .limit(200)
        )

        for scene in scenes:
            emb = scene.get("embedding")
            if emb is None:
                continue
            score = _cosine_similarity(query_embedding, np.array(emb))
            results.append({
                "type": "scene",
                "snapshot_id": scene.get("snapshot_id"),
                "camera": scene.get("camera", "webcam"),
                "timestamp": scene.get("timestamp"),
                "score": round(score, 4),
                "search_type": "semantic",
            })

        # ── Search event crop embeddings ──
        events_with_emb = list(
            mongodb.database.events.find(
                {"embedding": {"$exists": True}},
                {
                    "_id": 0,
                    "event_id": 1,
                    "event_type": 1,
                    "track_id": 1,
                    "class_name": 1,
                    "confidence": 1,
                    "camera": 1,
                    "timestamp": 1,
                    "embedding": 1,
                },
            )
            .sort("timestamp", -1)
            .limit(200)
        )

        for event in events_with_emb:
            emb = event.get("embedding")
            if emb is None:
                continue
            score = _cosine_similarity(query_embedding, np.array(emb))
            results.append({
                "type": "event",
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "track_id": event.get("track_id"),
                "class_name": event.get("class_name"),
                "confidence": event.get("confidence"),
                "camera": event.get("camera", "webcam"),
                "timestamp": event.get("timestamp"),
                "score": round(score, 4),
                "search_type": "semantic",
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Hybrid search combining Keyword matching + Semantic similarity.
        Fuses both score channels to return the top-K relevant results.
        """
        k = top_k or settings.SEARCH_TOP_K

        keyword_hits = self.keyword_search(query, limit=20)
        semantic_hits = self.semantic_search(query, top_k=20)

        # Merge and deduplicate by key (event_id or snapshot_id)
        merged = {}

        for hit in keyword_hits:
            key = hit.get("event_id") or hit.get("snapshot_id")
            if key:
                # 40% weight to keyword match
                merged[key] = {**hit, "score": round(hit["score"] * 0.4, 4)}

        for hit in semantic_hits:
            key = hit.get("event_id") or hit.get("snapshot_id")
            if key:
                if key in merged:
                    # Combined score: 40% keyword + 60% semantic
                    combined = merged[key]["score"] + (hit["score"] * 0.6)
                    merged[key]["score"] = round(combined, 4)
                    merged[key]["search_type"] = "hybrid"
                else:
                    # 60% weight to pure semantic
                    merged[key] = {**hit, "score": round(hit["score"] * 0.6, 4)}

        final_results = list(merged.values())
        final_results.sort(key=lambda r: r["score"], reverse=True)
        return final_results[:k]


search_service = SearchService()
