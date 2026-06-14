from fastapi import FastAPI

from src.api.routes_health import router as health_router
from src.api.routes_music import router as music_router
from src.api.routes_store import router as store_router

app = FastAPI(
    title="DB2 Proyecto 2 API",
    description="Hito 1: infraestructura base, PostgreSQL con pgvector y endpoints mock.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(music_router)
app.include_router(store_router)


@app.get("/")
def root():
    return {
        "message": "DB2 Proyecto 2 API",
        "module": "Docker + PostgreSQL + pgvector + FastAPI",
        "status": "mock endpoints ready",
    }
