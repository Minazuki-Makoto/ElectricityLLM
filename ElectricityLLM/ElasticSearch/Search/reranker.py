from typing import Any
import os
from pathlib import Path

import torch
from FlagEmbedding import FlagReranker


MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "LLM-data" / "Models" / "reranker-model"),
)

model = FlagReranker(
    model_name_or_path=MODEL_PATH,
    use_fp16=False,
    devices=os.getenv("RERANKER_MODEL_DEVICE", "cpu"),
)


def _compute_score_compat(
    pairs: list[list[str]],
    normalize: bool = True,
    batch_size: int = 8,
) -> list[float]:
    """Score text pairs without the tokenizer API removed in Transformers 5."""
    scores: list[float] = []
    device = next(model.model.parameters()).device

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        queries = [pair[0] for pair in batch]
        passages = [pair[1] for pair in batch]
        inputs = model.tokenizer(
            queries,
            text_pair=passages,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            logits = model.model(
                **inputs,
                return_dict=True,
            ).logits.view(-1).float()
        if normalize:
            logits = torch.sigmoid(logits)
        scores.extend(logits.cpu().tolist())

    return scores

def rerank_by_matched_queries(
    results: list[dict[str, Any]],
    fallback_query: str,
    threshold_score:float = 0.20
) -> list[dict[str, Any]]:
    """Score each candidate against the subqueries that retrieved it."""
    prepared_results = [result.copy() for result in results if result.get("chunk_id")]
    pairs: list[list[str]] = []
    pair_locations: list[tuple[int, str]] = []

    for result_index, result in enumerate(prepared_results):
        text = str(result.get("text", "")).strip()
        if not text:
            continue
        matched_queries = result.get("matched_queries") or [fallback_query]
        for matched_query in matched_queries:
            pairs.append([str(matched_query), text])
            pair_locations.append((result_index, str(matched_query)))

    if not pairs:
        return []

    scores = _compute_score_compat(pairs, normalize=True)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if not isinstance(scores, (list, tuple)):
        scores = [scores]

    for result in prepared_results:
        result["rerank_scores"] = {}

    for (result_index, matched_query), score in zip(pair_locations, scores):
        prepared_results[result_index]["rerank_scores"][matched_query] = float(score)

    valid_results = []
    for result in prepared_results:
        query_scores = result.get("rerank_scores", {})
        if not query_scores:
            continue
        best_query, best_score = max(query_scores.items(), key=lambda item: item[1])
        if best_score >= threshold_score:
            result["best_matched_query"] = best_query
            result["rerank_score"] = float(best_score)
            valid_results.append(result)

    return sorted(
        valid_results,
        key=lambda item: item.get("rerank_score", 0.0),
        reverse=True,
    )
