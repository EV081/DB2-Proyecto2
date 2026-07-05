from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_fashion import router as fashion_router
from src.api.routes_health import router as health_router
from src.api.routes_music import router as music_router

app = FastAPI(
    title="DB2 Proyecto 2 API",
    description="Hito 3: busqueda multimodal con SPIMI vs pgvector vs GIN/GiST.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(music_router)
app.include_router(fashion_router)


@app.get("/")
def root():
    return {
        "message": "DB2 Proyecto 2 API",
        "apps": ["music", "fashion"],
        "engines": ["spimi", "pgvector", "gin", "gist"],
    }
