from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.camera import router as camera_router
from app.api.routes.chat import router as chat_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.events import router as events_router
from app.api.routes.system import router as system_router
from app.api.routes.video import router as video_router
from app.config.settings import settings
from app.database.mongodb import mongodb
from app.services.camera.camera_manager import camera_manager
from app.services.event_service import event_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongodb.connect()
    event_service.init_indexes()
    yield
    camera_manager.stop()
    await mongodb.disconnect()


app = FastAPI(
    title="Eagle AI",
    description="AI-Powered Intelligent Surveillance Assistant",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes.visual import router as visual_router

app.include_router(system_router)
app.include_router(camera_router)
app.include_router(events_router)
app.include_router(embeddings_router)
app.include_router(video_router)
app.include_router(chat_router)
app.include_router(visual_router)


@app.get("/")
async def root():
    return {"message": "Eagle AI Backend Running"}
