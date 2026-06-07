def search_with_pgvector_mock(query: str = "demo") -> list[dict]:
    """Mock pgvector search placeholder. Real implementation belongs to Hito 2 or Hito 3."""
    return [
        {"id": 1, "title": "Resultado pgvector demo 1", "score": 0.93, "query": query},
        {"id": 2, "title": "Resultado pgvector demo 2", "score": 0.88, "query": query},
    ]


def search_with_gin_mock(query: str = "demo") -> list[dict]:
    """Mock GIN search placeholder. Real implementation belongs to Hito 2 or Hito 3."""
    return [
        {"id": 1, "title": "Resultado GIN demo 1", "score": 0.9, "query": query},
        {"id": 2, "title": "Resultado GIN demo 2", "score": 0.84, "query": query},
    ]


def search_with_gist_mock(query: str = "demo") -> list[dict]:
    """Mock GiST search placeholder. Real implementation belongs to Hito 2 or Hito 3."""
    return [
        {"id": 1, "title": "Resultado GiST demo 1", "score": 0.89, "query": query},
        {"id": 2, "title": "Resultado GiST demo 2", "score": 0.82, "query": query},
    ]
