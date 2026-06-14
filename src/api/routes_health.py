from fastapi import APIRouter

from src.db.database import check_database_status

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "db2-proyecto2-api"}


@router.get("/api/db/status")
def database_status():
    return check_database_status()
