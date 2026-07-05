from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from src.api.search_service import search_fashion_desc, search_fashion_image
from src.db.database import get_session
from src.db.models import Product

router = APIRouter(prefix="/api/fashion", tags=["fashion"])

DescEngine = Literal["spimi", "gin", "gist"]
ImageEngine = Literal["spimi", "pgvector"]


@router.get("/search/description")
def search_by_description(
    q: str = Query(..., description="Texto de busqueda"),
    engine: DescEngine = Query("spimi"),
    k: int = Query(10, ge=1, le=100),
):
    try:
        return search_fashion_desc(query_text=q, engine=engine, k=k)
    except LookupError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"BD no disponible: {e.__class__.__name__}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    engine: ImageEngine = Query("spimi"),
    k: int = Query(10, ge=1, le=100),
):
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return search_fashion_image(image_path=tmp_path, engine=engine, k=k)
    except LookupError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"BD no disponible: {e.__class__.__name__}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/media/{item_id}")
def get_fashion_media(item_id: str):
    with get_session() as session:
        product = None
        if item_id.isdigit():
            product = session.get(Product, int(item_id))
        if product is None:
            stmt = select(Product).where(Product.image_path.ilike(f"%{item_id}.%"))
            product = session.exec(stmt).first()
        if product is None or not product.image_path:
            raise HTTPException(status_code=404, detail="Producto o imagen no encontrada")
        image_path = Path(product.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado en disco")
        return FileResponse(image_path)