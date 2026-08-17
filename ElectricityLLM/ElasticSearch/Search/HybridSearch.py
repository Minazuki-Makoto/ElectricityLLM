"""Fuse keyword and vector retrieval results into one ranked chunk list."""
from typing import Any, Dict, Iterable, List

from .SearchFuction import search_functions


def _flatten_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept both flat results and the current ``{chunk_id: data}`` format."""
    flattened: List[Dict[str, Any]] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue

        if "chunk_id" in result:
            item = dict(result)
            item["score"] = float(item.get("score", 0.0) or 0.0)
            flattened.append(item)
            continue

        for chunk_id, data in result.items():
            if chunk_id == "score" or not isinstance(data, dict):
                continue
            item = dict(data)
            item["chunk_id"] = item.get("chunk_id") or item.get("id") or chunk_id
            item["score"] = float(item.get("score", 0.0) or 0.0)
            flattened.append(item)

    return flattened


def _add_weighted_hybrid_results(
    gathered: Dict[str, Dict[str, Any]],
    results: Iterable[Dict[str, Any]],
    weight: float,
    score_name: str,
    raw_score_weight: float,
    rrf_score_weight: float,
    rrf_k: int = 20,
    is_document_match: bool = False,
) -> None:
    """Fuse normalized retrieval strength and reciprocal-rank evidence."""

    document_ranks: Dict[str, int] = {}
    for result_rank, result in enumerate(_flatten_results(results), start=1):
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue


        if chunk_id not in gathered:
            gathered[chunk_id] = {
                **result,
                "chunk_id": chunk_id,
                "score": 0.0,
                "score_details": {},
            }

        if is_document_match:
            document_title = str(result.get("title", ""))
            if document_title not in document_ranks:
                document_ranks[document_title] = len(document_ranks) + 1
            rank = document_ranks[document_title]
        else:
            rank = result_rank

        raw_score = float(result.get("score", 0.0) or 0.0)
        rrf_score = 1.0 / (rrf_k + rank)
        normalized_rrf_score = (rrf_k + 1) * rrf_score

        combined_score = (
            raw_score * raw_score_weight
            + normalized_rrf_score * rrf_score_weight
        )
        weighted_score = combined_score * weight
        gathered[chunk_id]["score"] += weighted_score
        gathered[chunk_id]["score_details"][score_name] = {
            "rank": rank,
            "raw_score": raw_score,
            "raw_score_weight": raw_score_weight,
            "rrf_k": rrf_k,
            "rrf_score": rrf_score,
            "normalized_rrf_score": normalized_rrf_score,
            "rrf_score_weight": rrf_score_weight,
            "combined_score": combined_score,
            "channel_weight": weight,
            "weighted_score": weighted_score,
        }


def hybrid_search(
    elasticsearch_uris,
    elasticsearch_username,
    elasticsearch_password,
    elastic_index,
    query,
    document_title_weight,
    title_weight,
    text_weight,
    vector_weight,
    description_text_weight,
    topk,
    score_threshold=None,
    raw_score_weight=0.8,
    rrf_score_weight=0.2,
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk 必须是正整数")
    if score_threshold is not None:
        score_threshold = float(score_threshold)
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold 必须在 0 到 1 之间")

    raw_score_weight = float(raw_score_weight)
    rrf_score_weight = float(rrf_score_weight)
    if raw_score_weight < 0 or rrf_score_weight < 0:
        raise ValueError("原始分数和 RRF 分数权重不能为负数")
    fusion_weight_sum = raw_score_weight + rrf_score_weight
    if fusion_weight_sum <= 0:
        raise ValueError("原始分数和 RRF 分数至少需要一个大于 0 的权重")
    raw_score_weight /= fusion_weight_sum
    rrf_score_weight /= fusion_weight_sum

    if float(document_title_weight) < 0:
        raise ValueError("检索权重不能为负数")

    weights = {
        "document_title":float(document_title_weight),
        "title": float(title_weight),
        "text": float(text_weight),
        "vector": float(vector_weight),
        "description": float(description_text_weight),
    }

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("检索权重不能为负数")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("至少需要一个大于 0 的检索权重")
    weights = {name: weight / total_weight for name, weight in weights.items()}

    search = search_functions(
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        elastic_index,
        query.strip(),
    )

    # Each retrieval channel needs more candidates than the final top-k so
    # useful chunks appearing in only one channel still have a chance to rank.
    candidate_k = max(topk * 5, 50)
    document_title_results = search.document_title_match(topk=candidate_k)
    title_results = search.title_match(topk=candidate_k)
    text_results = search.keyword_match(topk=candidate_k)
    text_vector_results = search.vector_match("text_embedding", topk=candidate_k)
    description_vector_results = search.vector_match(
        "description_embedding",
        topk=candidate_k,
    )

    description_text_results = search.description_match(topk=candidate_k)



    gathered: Dict[str, Dict[str, Any]] = {}

    def add_channel(results, weight, score_name, is_document_match=False):
        _add_weighted_hybrid_results(
            gathered=gathered,
            results=results,
            weight=weight,
            score_name=score_name,
            raw_score_weight=raw_score_weight,
            rrf_score_weight=rrf_score_weight,
            is_document_match=is_document_match,
        )

    add_channel(
        document_title_results,
        weights["document_title"],
        "document_title",
        is_document_match=True,
    )

    add_channel(title_results, weights["title"], "title")
    add_channel(text_results, weights["text"], "text")
    add_channel(text_vector_results, weights["vector"] * 0.3, "text_vector")

    add_channel(
        description_vector_results,
        weights["vector"] * 0.7,
        "description_vector",
    )
    add_channel(
        description_text_results,
        weights["description"],
        "description_text",
    )
    ranked_results = sorted(
        gathered.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    for result in ranked_results:
        result["query_keywords"] = list(search.keywords)
        result["weighted_query_keywords"] = dict(search.weighted_keywords)

    # Both fusion components are normalized to 0..1 before channel weighting.
    # By default return strict Top-K; callers may explicitly add a threshold.
    if score_threshold is not None:
        ranked_results = [
            result
            for result in ranked_results
            if result["score"] >= score_threshold
        ]
    return ranked_results[:topk]









