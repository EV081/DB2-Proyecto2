from fastapi import APIRouter

router = APIRouter(prefix="/api/music", tags=["music"])


@router.get("/search")
def search_music(query: str = "demo"):
    return {
        "query": query,
        "modality": "audio_text",
        "engine": "mock",
        "results": [
            {"id": 1, "title": "Cancion demo 1", "score": 0.92},
            {"id": 2, "title": "Cancion demo 2", "score": 0.87},
        ],
    }
