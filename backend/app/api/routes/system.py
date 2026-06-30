from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter(
    prefix="/system",
    tags=["System"]
)


@router.get("/status")
async def system_status():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "backend": "Running",
        "database": "Connected",
        "status": "Healthy"
    }


@router.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "Running",
        "message": "Welcome to Eagle AI Backend 🚀"
    }