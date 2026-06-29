from fastapi import FastAPI

app = FastAPI(
    title="Eagle AI",
    description="AI-Powered Intelligent Surveillance Assistant",
    version="1.0.0",
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint to verify that the backend is running.
    """
    return {
        "project": "Eagle AI",
        "version": "1.0.0",
        "status": "Running",
        "message": "Welcome to Eagle AI Backend 🚀"
    }