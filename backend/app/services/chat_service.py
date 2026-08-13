from datetime import datetime
import re
import httpx

from app.config.settings import settings
from app.database.mongodb import mongodb
from app.services.search_service import search_service


def _format_timestamp(ts) -> str:
    """Format raw datetime/ISO string into concise, human-readable timestamp."""
    if not ts:
        return "Unknown Time"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return ts
    if isinstance(ts, datetime):
        return ts.strftime("%b %d, %H:%M:%S")
    return str(ts)


def _transform_query_for_clip(query: str) -> str:
    """
    Transform conversational user questions into descriptive image prompts.
    CLIP models perform significantly better with image captions than questions.
    """
    q = query.strip().lower()

    # Remove question phrasing
    q = re.sub(r"^(is|are|was|were|can you see|show me|find|look for|did anyone|has anyone)\s+(there|a|an|any)?\s*", "", q)
    q = re.sub(r"\?$", "", q)

    if not q:
        return query

    return f"a surveillance camera photo of {q}"


SYSTEM_PROMPT = """You are Eagle AI, an intelligent surveillance assistant.
You answer questions about events observed by security cameras.

Below is the relevant context retrieved from the surveillance database.

RECENT EVENTS:
{events_context}

MATCHED RESULTS (ranked by relevance):
{search_context}

RULES:
- Answer using ONLY the information provided above.
- Always mention timestamps and track IDs when relevant.
- Use concise, readable time formats (e.g., "Jul 01, 13:36:52").
- If the information is not available in the context, say so honestly.
- Be concise and precise. Security operators need clear, actionable answers.
"""


class ChatService:
    """
    RAG (Retrieval-Augmented Generation) chat service supporting both:
    1. Local LLM via Ollama (llama3.2)
    2. Cloud LLM via Groq API (llama-3.1-8b-instant, gemma2-9b-it)
    """

    def _get_recent_events(self, session_id: str | None = None, limit: int = 10) -> list[dict]:
        """Fetch the most recent events from MongoDB, optionally filtered by session."""
        if mongodb.database is None:
            return []

        query = {}
        if session_id:
            query["session_id"] = session_id

        try:
            return list(
                mongodb.database.events.find(
                    query,
                    {"_id": 0, "embedding": 0},
                ).sort("timestamp", -1).limit(limit)
            )
        except Exception as exc:
            print(f"Chat service events query note: {exc}")
            return []

    def _format_events_context(self, events: list[dict]) -> str:
        """Format events list into a readable string for the LLM."""
        if not events:
            return "No events recorded yet."

        lines = []
        for ev in events:
            ts = _format_timestamp(ev.get("timestamp"))
            event_type = ev.get("event_type", "UNKNOWN")
            track_id = ev.get("track_id", "?")
            class_name = ev.get("class_name", "")
            confidence = ev.get("confidence")
            attrs = ev.get("attributes")

            line = f"T{track_id} {class_name} @ {ts}"
            if attrs:
                line += f" ({', '.join(attrs)})"
            if confidence:
                line += f" {confidence:.0%}"
            lines.append(line)

        return "\n".join(lines)

    def _format_search_context(self, results: list[dict]) -> str:
        """Format hybrid search results into a readable string for the LLM."""
        if not results:
            return "No matching visual or keyword results found."

        lines = []
        for i, r in enumerate(results, 1):
            ts = _format_timestamp(r.get("timestamp"))
            score = r.get("score", 0)

            if r.get("type") == "scene":
                caption = r.get("caption") or r.get("summary") or "Visual camera frame captured"
                category = r.get("category", "normal")
                lines.append(
                    f"  {i}. [Scene Match ({category})] at {ts} (Relevance: {score:.0%}) — Description: \"{caption}\""
                )
            else:
                event_type = r.get("event_type", "UNKNOWN")
                track_id = r.get("track_id", "?")
                class_name = r.get("class_name", "")
                attrs = r.get("attributes")
                attrs_str = f" ({', '.join(attrs)})" if attrs else ""
                lines.append(
                    f"  {i}. [{event_type}] Track #{track_id} {class_name}{attrs_str} "
                    f"at {ts} (Relevance: {score:.0%})"
                )

        return "\n".join(lines)

    def _build_messages(
        self,
        question: str,
        history: list[dict],
        search_results: list[dict],
        recent_events: list[dict],
    ) -> list[dict]:
        """Construct system + history + question message array."""
        system_content = SYSTEM_PROMPT.format(
            events_context=self._format_events_context(recent_events),
            search_context=self._format_search_context(search_results),
        )

        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        return messages

    def ask(self, question: str, history: list[dict] | None = None, session_id: str | None = None) -> dict:
        """
        Full RAG pipeline: retrieve context → build prompt → call LLM.
        """
        history = history or []
        # Trim history to last 4 turns (2 user, 2 assistant) to save tokens
        if len(history) > 4:
            history = history[-4:]

        # Step 1: Hybrid search (Keyword + Semantic)
        # Note: We limit search to top 3 for token optimization
        search_results = search_service.search(question, top_k=3)

        # Filter search results if session_id is provided
        if session_id:
            search_results = [r for r in search_results if r.get("session_id") == session_id or not r.get("session_id")]

        # Step 2: Fetch recent events for temporal context
        recent_events = self._get_recent_events(session_id=session_id, limit=10)

        # Step 3: Build messages
        messages = self._build_messages(
            question, history, search_results, recent_events
        )

        # Step 4: Call selected LLM Provider (Groq or Ollama)
        provider = (settings.LLM_PROVIDER or "ollama").lower()

        if provider == "groq":
            answer = self._call_groq(messages)
        else:
            answer = self._call_ollama(messages)

        # Step 5: Format sources and visual refs
        sources = []
        visual_refs = []
        
        for r in search_results:
            source = {k: v for k, v in r.items() if k != "embedding"}
            sources.append(source)
            
            # Extract visual evidence refs if they have frame_number
            if r.get("type") == "event" and r.get("frame_number") is not None:
                evt_id = r.get("event_id")
                cls_name = r.get("class_name", "object")
                ts_sec = r.get("timestamp_sec", 0)
                
                # Format MM:SS
                m, s = divmod(int(ts_sec), 60)
                ts_display = f"{m:02d}:{s:02d}"
                
                visual_refs.append({
                    "event_id": evt_id,
                    "label": f"{cls_name} detected",
                    "timestamp_sec": ts_sec,
                    "timestamp_display": ts_display,
                    "frame_url": f"/visual/frame/{evt_id}",
                    "clip_url": f"/visual/clip/{evt_id}"
                })

        return {
            "answer": answer,
            "sources": sources,
            "visual_refs": visual_refs,
            "provider": provider,
        }

    def _call_groq(self, messages: list[dict]) -> str:
        """Call Groq API (cloud LLM inference)."""
        if not settings.GROQ_API_KEY:
            return (
                "⚠️ `GROQ_API_KEY` is not set in `.env`.\n"
                "To use Groq Cloud LLM, add `GROQ_API_KEY=gsk_...` to your `.env` file."
            )

        model = settings.LLM_MODEL if "llama-3" in settings.LLM_MODEL or "gemma" in settings.LLM_MODEL else "llama-3.1-8b-instant"

        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ Groq API Error ({response.status_code}): {response.text}"
        except httpx.TimeoutException:
            return "⚠️ Groq API request timed out."
        except Exception as exc:
            return f"⚠️ Groq API connection error: {exc}"

    def _call_ollama(self, messages: list[dict]) -> str:
        """Call Ollama LLM (local edge inference)."""
        try:
            response = httpx.post(
                f"{settings.OLLAMA_HOST}/api/chat",
                json={
                    "model": settings.LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                },
                timeout=180.0,
            )
        except httpx.ConnectError:
            return (
                f"⚠️ Cannot connect to Ollama at `{settings.OLLAMA_HOST}`.\n"
                "Make sure Ollama is running (`ollama serve`) or switch to `LLM_PROVIDER=groq` in `.env`."
            )
        except httpx.TimeoutException:
            return (
                "⚠️ Ollama timed out loading model. Please try again."
            )
        except Exception as exc:
            return f"⚠️ Connection error: {exc}"

        if response.status_code != 200:
            try:
                error_msg = response.json().get("error", response.text)
            except Exception:
                error_msg = response.text

            if "memory" in error_msg.lower():
                return (
                    f"⚠️ **Out of Memory**: {error_msg}\n\n"
                    "• Stop the live camera stream first\n"
                    "• Or switch to `LLM_PROVIDER=groq` in `.env` for instant cloud LLM response."
                )
            elif response.status_code == 404:
                return f"⚠️ Model `{settings.LLM_MODEL}` not found in Ollama. Run `ollama pull {settings.LLM_MODEL}`."
            else:
                return f"⚠️ Ollama error ({response.status_code}): {error_msg}"

        try:
            return response.json()["message"]["content"]
        except (KeyError, Exception) as exc:
            return f"⚠️ Failed to parse response: {exc}"


chat_service = ChatService()
