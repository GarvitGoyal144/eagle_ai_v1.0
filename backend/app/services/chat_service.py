import httpx

from app.config.settings import settings
from app.database.mongodb import mongodb
from app.services.search_service import search_service


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
- If the information is not available in the context, say so honestly.
- Be concise and precise. Security operators need clear, actionable answers.
- Format timestamps in a human-readable way.
"""


class ChatService:
    """
    RAG (Retrieval-Augmented Generation) chat service supporting both:
    1. Local LLM via Ollama (llama3.2)
    2. Cloud LLM via Groq API (llama-3.1-8b-instant, gemma2-9b-it)
    """

    def _get_recent_events(self, limit: int = 20) -> list[dict]:
        """Fetch the most recent events from MongoDB."""
        if mongodb.database is None:
            return []

        return list(
            mongodb.database.events.find(
                {},
                {"_id": 0, "embedding": 0},
            ).sort("timestamp", -1).limit(limit)
        )

    def _format_events_context(self, events: list[dict]) -> str:
        """Format events list into a readable string for the LLM."""
        if not events:
            return "No events recorded yet."

        lines = []
        for ev in events:
            ts = ev.get("timestamp", "unknown time")
            event_type = ev.get("event_type", "UNKNOWN")
            track_id = ev.get("track_id", "?")
            class_name = ev.get("class_name", "")
            confidence = ev.get("confidence")

            line = f"- [{ts}] {event_type}: Track #{track_id}"
            if class_name:
                line += f" ({class_name})"
            if confidence:
                line += f" confidence={confidence:.0%}"
            lines.append(line)

        return "\n".join(lines)

    def _format_search_context(self, results: list[dict]) -> str:
        """Format hybrid search results into a readable string."""
        if not results:
            return "No matching results found."

        lines = []
        for i, r in enumerate(results, 1):
            ts = r.get("timestamp", "unknown time")
            score = r.get("score", 0)

            if r["type"] == "scene":
                lines.append(
                    f"  {i}. [Scene Snapshot] {ts} — relevance: {score:.2f}"
                )
            else:
                event_type = r.get("event_type", "UNKNOWN")
                track_id = r.get("track_id", "?")
                class_name = r.get("class_name", "")
                lines.append(
                    f"  {i}. [{event_type}] Track #{track_id} {class_name} "
                    f"at {ts} — relevance: {score:.2f}"
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

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """
        Full RAG pipeline: retrieve context → build prompt → call LLM.
        """
        history = history or []

        # Step 1: Hybrid search (Keyword + Semantic)
        search_results = search_service.search(question)

        # Step 2: Fetch recent events for temporal context
        recent_events = self._get_recent_events(limit=20)

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

        # Step 5: Format sources
        sources = [
            {k: v for k, v in r.items() if k != "embedding"}
            for r in search_results
        ]

        return {
            "answer": answer,
            "sources": sources,
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
