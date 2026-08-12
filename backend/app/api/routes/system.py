from fastapi import APIRouter

from app.config.settings import settings
from app.database.mongodb import mongodb

router = APIRouter(
    prefix="/system",
    tags=["System"]
)


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Docker and Hugging Face Spaces.
    Returns HTTP 200 when backend is alive and DB is reachable.
    """
    db_ok = mongodb.is_connected()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


@router.get("/status")
async def system_status():
    """Full system status with real DB ping."""
    db_ok = mongodb.is_connected()
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "backend": "Running",
        "database": "Connected" if db_ok else "Unreachable",
        "status": "Healthy" if db_ok else "Degraded",
    }


@router.get("")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "Running",
        "message": "Welcome to Eagle AI Backend 🚀"
    }