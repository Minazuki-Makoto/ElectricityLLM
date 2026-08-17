import re
from typing import Any, Dict, List, Optional

from ..Elastic_base import link
from ..embedding.query_embedded import analyze_query


class search_functions:
    def __init__(
        self,
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        elastic_index,
        query,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")

        self.es = link(
            elasticsearch_uris,
            elasticsearch_username,
            elasticsearch_password,
        )
        self.index = elastic_index
        self.query = query.strip()
        self.embedded_query, self.weighted_keywords = analyze_query(self.query)
        self.keywords = [keyword for keyword, _ in self.weighted_keywords]
        self.keyword_query = " ".join(self.keywords)

    @staticmethod
    def _clean_title_query(query: str) -> str:
        cleaned = re.sub(r"[？?！!，,。；;：:\s]+", "", query.strip())
        for phrase in (
            "请问",
            "请介绍一下",
            "介绍一下",
            "讲一下",
            "什么是",
            "有哪些",
            "有什么",
        ):
            cleaned = cleaned.replace(phrase, "")
        return cleaned or query.strip()

    @staticmethod
    def _parse_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id") or hit.get("_id")
            if not chunk_id:
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": source.get("chunk_index"),
                    "score": float(hit.get("_score", 0.0) or 0.0),
                    "title": source.get("title", ""),
                    "chapter_title": source.get("chapter_title", ""),
                    "section_title": source.get("section_title", ""),
                    "third_title": source.get("third_title", ""),
                    "fourth_title": source.get("fourth_title", ""),
                    "text": source.get("text", ""),
                    "description": source.get("description", ""),
                }
            )
        return results

    def keyword_match(self, query=None, index=None, topk: int = 5):
        index = index or self.index
        query = query or self.query
        should_queries = [
            {"match": {"text": {"query": query, "boost": 1.0}}},
        ]
        if self.keyword_query:
            should_queries.append(
                {
                    "match": {
                        "text": {
                            "query": self.keyword_query,
                            "operator": "or",
                            "boost": 1.2,
                        }
                    }
                }
            )
        response = self.es.search(
            index=index,
            query={
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1,
                }
            },
            size=topk,
            source_excludes=["text_embedding", "description_embedding"],
        )
        results = self.normalization(self._parse_hits(response))
        return results

    def title_match(self, query=None, index=None, topk: int = 5):
        index = index or self.index
        original_query = query or self.query
        cleaned_query = self._clean_title_query(original_query)
        fields = [
            "chapter_title^5",
            "section_title^3",
            "third_title^1.5",
            "fourth_title",
        ]

        response = self.es.search(
            index=index,
            query={
                "multi_match": {
                    "query": original_query,
                    "fields": fields,
                    "type": "best_fields",
                    "minimum_should_match": "60%",
                }
            },
            size=topk,
            source_excludes=["text_embedding", "description_embedding"],
        )
        results = self.normalization(self._parse_hits(response))
        return results

    def document_title_match(self, query=None, index=None, topk: int = 5):
        index = index or self.index
        query = query or self.query
        response = self.es.search(
            index=index,
            query={"match": {"title": {"query": query}}},
            size=topk,
            source_excludes=["text_embedding", "description_embedding"],
        )

        results = self.normalization(self._parse_hits(response))

        return results

    def vector_match(
        self,
        embedding_mode: str,
        query_vector: Optional[List[float]] = None,
        index=None,
        topk: int = 5,
    ):
        if embedding_mode not in {"text_embedding", "description_embedding"}:
            raise ValueError("embedding_mode 必须是 text_embedding 或 description_embedding")
        index = index or self.index
        if query_vector is None:
            query_vector = self.embedded_query
        response = self.es.search(
            index=index,
            knn={
                "field": embedding_mode,
                "query_vector": query_vector,
                "k": topk,
                "num_candidates": max(topk * 10, 50),
            },
            size=topk,
            source_excludes=["text_embedding", "description_embedding"],
        )
        results = self.normalization(self._parse_hits(response))
        return results

    def description_match(self, index=None, query=None, topk: int = 5):
        index = index or self.index
        query = query or self.query
        should_queries = [
            {"match": {"description": {"query": query, "boost": 1.0}}},
        ]
        if self.keyword_query:
            should_queries.append(
                {
                    "match": {
                        "description": {
                            "query": self.keyword_query,
                            "operator": "or",
                            "boost": 1.2,
                        }
                    }
                }
            )
        response = self.es.search(
            index=index,
            query={
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1,
                }
            },
            size=topk,
            source_excludes=["text_embedding", "description_embedding"],
        )
        results = self.normalization(self._parse_hits(response))
        return results

    @staticmethod
    def normalization(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        scores = [float(result.get("score", 0.0) or 0.0) for result in results]
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            normalized_score = 1.0 if max_score != 0 else 0.0
            for result in results:
                result["score"] = normalized_score
            return results

        score_range = max_score - min_score
        for result in results:
            score = float(result.get("score", 0.0) or 0.0)
            result["score"] = (score - min_score) / score_range

        return results

