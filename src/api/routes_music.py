from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from src.api.search_service import search_music_audio, search_music_lyrics
from src.db.database import get_session
from src.db.models import Song

router = APIRouter(prefix="/api/music", tags=["music"])

LyricsEngine = Literal["spimi", "gin", "gist"]
AudioEngine = Literal["spimi", "pgvector"]


@router.get("/search/lyrics")
def search_by_lyrics(
    q: str = Query(..., description="Texto de busqueda"),
    engine: LyricsEngine = Query("spimi"),
    k: int = Query(10, ge=1, le=100),
):
    try:
        return search_music_lyrics(query_text=q, engine=engine, k=k)
    except LookupError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"BD no disponible: {e.__class__.__name__}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search/audio")
async def search_by_audio(
    file: UploadFile = File(...),
    engine: AudioEngine = Query("spimi"),
    k: int = Query(10, ge=1, le=100),
):
    suffix = Path(file.filename or "query.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return search_music_audio(audio_path=tmp_path, engine=engine, k=k)
    except LookupError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"BD no disponible: {e.__class__.__name__}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/media/{item_id}")
def get_music_media(item_id: str):
    with get_session() as session:
        song = None
        if item_id.isdigit():
            song = session.get(Song, int(item_id))
        if song is None:
            stmt = select(Song).where(Song.audio_path.ilike(f"%{item_id}.%"))
            song = session.exec(stmt).first()
        if song is None or not song.audio_path:
            raise HTTPException(status_code=404, detail="Cancion o audio no encontrado")
        audio_path = Path(song.audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado en disco")
        return FileResponse(audio_path)