import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Support both PyCharm's "Run file" action and ``python -m`` execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ElasticSearch.Search.HybridSearch import hybrid_search
from ElasticSearch.Search.reranker import rerank_by_matched_queries


def split_query(query: str, max_subqueries: int = 5) -> List[str]:
    """Split only at explicit sentence boundaries and preserve query order."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    if max_subqueries <= 0:
        raise ValueError("max_subqueries 必须是正整数")

    subqueries: List[str] = []
    seen = set()
    for part in re.split(r"[？?。；;！!\n]+", query.strip()):
        normalized = part.strip(" ，,")
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        subqueries.append(normalized)
        if len(subqueries) >= max_subqueries:
            break

    return subqueries or [query.strip()]


def _merge_candidates(
    candidate_groups: List[List[Dict[str, Any]]],
    subqueries: List[str],
) -> List[Dict[str, Any]]:
    """Deduplicate chunks while retaining their strongest retrieval evidence."""
    candidates: Dict[str, Dict[str, Any]] = {}

    for subquery, results in zip(subqueries, candidate_groups):
        for result in results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue

            score = float(result.get("score", 0.0) or 0.0)
            if chunk_id not in candidates:
                item = result.copy()
                item["matched_queries"] = [subquery]
                item["subquery_scores"] = {subquery: score}
                item["subquery_keywords"] = {
                    subquery: list(result.get("query_keywords", []))
                }
                candidates[chunk_id] = item
                continue

            item = candidates[chunk_id]
            item["subquery_scores"][subquery] = score
            item["subquery_keywords"][subquery] = list(
                result.get("query_keywords", [])
            )
            if subquery not in item["matched_queries"]:
                item["matched_queries"].append(subquery)

            # Retain the strongest hybrid evidence without summing duplicates.
            if score > float(item.get("score", 0.0) or 0.0):
                preserved_queries = item["matched_queries"]
                preserved_scores = item["subquery_scores"]
                preserved_keywords = item["subquery_keywords"]
                item.update(result)
                item["matched_queries"] = preserved_queries
                item["subquery_scores"] = preserved_scores
                item["subquery_keywords"] = preserved_keywords

    return list(candidates.values())


def _balanced_topk(
    results: List[Dict[str, Any]],
    subqueries: List[str],
    topk: int,
) -> List[Dict[str, Any]]:
    """Round-robin subquery rankings so one topic cannot crowd out another."""
    rankings: Dict[str, List[Dict[str, Any]]] = {}
    for subquery in subqueries:
        rankings[subquery] = sorted(
            [result for result in results if subquery in result.get("matched_queries", [])],
            key=lambda item: item.get("rerank_scores", {}).get(subquery, 0.0),
            reverse=True,
        )

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    positions = {subquery: 0 for subquery in subqueries}

    while len(selected) < topk:
        added = False
        for subquery in subqueries:
            ranking = rankings[subquery]
            while positions[subquery] < len(ranking):
                result = ranking[positions[subquery]]
                positions[subquery] += 1
                chunk_id = result.get("chunk_id")
                if chunk_id in selected_ids:
                    continue
                selected_ids.add(chunk_id)
                selected.append(result)
                added = True
                break
            if len(selected) >= topk:
                break
        if not added:
            break

    if len(selected) < topk:
        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id in selected_ids:
                continue
            selected_ids.add(chunk_id)
            selected.append(result)
            if len(selected) >= topk:
                break

    return selected


def full_find(
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
):
    """Retrieve each explicit question, merge candidates, then rerank once."""
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk 必须是正整数")

    subqueries = split_query(query)
    # Over-fetch because different subqueries can return the same chunks.
    per_query_topk = max(5, math.ceil(topk / len(subqueries)))

    candidate_groups: List[List[Dict[str, Any]]] = []
    for subquery in subqueries:
        candidate_groups.append(
            hybrid_search(
                elasticsearch_uris=elasticsearch_uris,
                elasticsearch_username=elasticsearch_username,
                elasticsearch_password=elasticsearch_password,
                elastic_index=elastic_index,
                query=subquery,
                document_title_weight=document_title_weight,
                title_weight=title_weight,
                text_weight=text_weight,
                vector_weight=vector_weight,
                description_text_weight=description_text_weight,
                topk=per_query_topk,
                score_threshold=score_threshold,
            )
        )

    candidates = _merge_candidates(candidate_groups, subqueries)
    reranked_results = rerank_by_matched_queries(candidates, query)
    return _balanced_topk(reranked_results, subqueries, topk)


def main():
    query = "讲讲风力发电的原理"
    results = full_find(
        elasticsearch_uris="https://localhost:9200",
        elasticsearch_username=os.environ["ELASTICSEARCH_USERNAME"],
        elasticsearch_password=os.environ["ELASTICSEARCH_PASSWORD"],
        elastic_index="electricity-infos",
        query=query,
        document_title_weight=0.10,
        title_weight=0.20,
        text_weight=0.02,
        vector_weight=0.60,
        description_text_weight=0.08,
        topk=30,
        score_threshold=0.30
    )

    print(f"最终返回 {len(results)} 条结果。")
    for index, result in enumerate(results, start=1):
        matched_queries = " | ".join(result.get("matched_queries", []))
        print(
            f"\n[{index}] chunk_id={result.get('chunk_id', '')}，"
            f"hybrid_score={result.get('score', 0.0):.4f}，"
            f"rerank_score={result.get('rerank_score', 0.0):.4f}"
        )
        print(f"matched_queries={matched_queries}")
        print(f"subquery_keywords={result.get('subquery_keywords', {})}")
        print(f"best_matched_query={result.get('best_matched_query', '')}")
        print(result.get("text", ""))
        print("=" * 60)

if __name__ == "__main__":
    main()
