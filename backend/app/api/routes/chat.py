from fastapi import APIRouter
from pydantic import BaseModel

from app.services.chat_service import chat_service
from app.services.search_service import search_service

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class SearchRequest(BaseModel):
    question: str
    top_k: int | None = None


@router.post("")
def chat(req: ChatRequest):
    """
    Send a natural language question about surveillance events.

    Uses the RAG pipeline:
    1. CLIP semantic search for relevant context
    2. Recent events for temporal context
    3. Ollama LLM for natural language reasoning

    Supports multi-turn conversation via the `history` field.
    """
    result = chat_service.ask(
        question=req.question,
        history=req.history,
    )
    return result


@router.post("/search")
def semantic_search(req: SearchRequest):
    """
    Semantic search only — no LLM.

    Returns ranked results matching the query by CLIP similarity.
    Useful for debugging or direct semantic retrieval.
    """
    results = search_service.search(
        query=req.question,
        top_k=req.top_k,
    )
    return {"results": results}
