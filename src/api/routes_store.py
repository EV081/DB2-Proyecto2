from fastapi import APIRouter

router = APIRouter(prefix="/api/store", tags=["store"])


@router.get("/search")
def search_store(query: str = "camisa azul"):
    return {
        "query": query,
        "modality": "image_text",
        "engine": "mock",
        "results": [
            {"id": 1, "title": "Producto demo 1", "score": 0.91},
            {"id": 2, "title": "Producto demo 2", "score": 0.85},
        ],
    }
