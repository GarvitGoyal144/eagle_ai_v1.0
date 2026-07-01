from fastapi import APIRouter

from app.database.mongodb import mongodb

router = APIRouter(
    prefix="/events",
    tags=["Events"]
)


@router.get("/")
async def get_events():

    events = list(
        mongodb.database.events.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(50)
    )

    return events