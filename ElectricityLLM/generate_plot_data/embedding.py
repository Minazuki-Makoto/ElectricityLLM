from __future__ import annotations

def embed_description(description: str) -> list[float]:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description不能为空")
    from ElasticSearch.embedding.query_embedded import embed
    return embed(description.strip())
