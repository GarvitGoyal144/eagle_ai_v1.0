from pydantic import BaseModel
from datetime import datetime


class EventResponse(BaseModel):
    event_id: str
    event_type: str
    track_id: int
    class_name: str
    confidence: float
    camera: str
    timestamp: datetime