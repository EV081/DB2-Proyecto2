import os
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/proyecto2",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_database_status() -> dict[str, Any]:
    """Return a simple PostgreSQL and pgvector availability status."""
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
