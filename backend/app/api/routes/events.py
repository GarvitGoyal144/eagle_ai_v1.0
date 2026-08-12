from fastapi import APIRouter, Query

from app.services.event_service import event_service

router = APIRouter(
    prefix="/events",
    tags=["Events"]
)


@router.get("")
def get_events(
    limit: int = Query(default=50, ge=1, le=500, description="Number of events to return"),
    offset: int = Query(default=0, ge=0, description="Number of events to skip (for pagination)"),
):
    """
    Fetch recent surveillance events with pagination support.

    - **limit**: max number of events to return (1–500, default 50)
    - **offset**: number of events to skip for pagination (default 0)
    """
    return event_service.get_events(limit=limit, offset=offset)