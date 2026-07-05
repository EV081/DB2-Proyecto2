import os
from contextlib import contextmanager
from urllib.parse import quote_plus
from typing import Any
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

load_dotenv()

def _build_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    database = os.getenv("POSTGRES_DB", "proyecto2")

    credentials = quote_plus(user)
    if password:
        credentials = f"{credentials}:{quote_plus(password)}"

    return f"postgresql://{credentials}@{host}:{port}/{database}"


DATABASE_URL = _build_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@contextmanager
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def check_database_connection() -> dict[str, Any]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            pgvector_available = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar()

        return {
            "database": "connected",
            "pgvector": "available" if pgvector_available else "unavailable",
        }
    except SQLAlchemyError as exc:
        return {
            "database": "error",
            "pgvector": "unknown",
            "detail": str(exc.__class__.__name__),
        }


def check_database_status() -> dict[str, Any]:
    return check_database_connection()
