import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

from FlagEmbedding import BGEM3FlagModel


model_path = os.getenv(
    "EMBEDDING_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "LLM-data" / "Models" / "bge-m3"),
)

model = BGEM3FlagModel(
    model_path,
    use_fp16=False,
    devices=os.getenv("EMBEDDING_MODEL_DEVICE", "cpu"),
)

_IGNORED_KEYWORDS = {
    "什么",
    "什么是",
    "怎么",
    "如何",
    "请问",
    "一下",
    "介绍",
    "说明",
    "又是",
    "？",
    "?",
    "。",
    "，",
    ",",
    "；",
    ";",
    "！",
    "!",
}


def _clean_token(token: str) -> str:
    token = token.replace("▁", "").replace("</w>", "").strip()
    return re.sub(r"^[\W_]+|[\W_]+$", "", token, flags=re.UNICODE)


def analyze_query(
    query: str,
    keyword_topk: int = 8,
    min_keyword_weight: float = 0.05,
) -> Tuple[List[float], List[Tuple[str, float]]]:
    """Generate one dense vector and weighted lexical keywords in one pass."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")
    if keyword_topk <= 0:
        raise ValueError("keyword_topk 必须是正整数")

    result = model.encode(
        sentences=[query.strip()],
        batch_size=1,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    token_weights: Dict[str, float] = model.convert_id_to_token(
        result["lexical_weights"][0]
    )
    normalized_tokens: List[Tuple[str, float]] = []
    for raw_token, raw_weight in token_weights.items():
        token = _clean_token(str(raw_token))
        weight = float(raw_weight)
        if (
            not token
            or token in _IGNORED_KEYWORDS
            or weight < min_keyword_weight
            or not re.search(r"[\w\u4e00-\u9fff]", token)
        ):
            continue
        normalized_tokens.append((token, weight))

    merged_tokens: List[Tuple[str, float]] = []
    index = 0
    while index < len(normalized_tokens):
        token, weight = normalized_tokens[index]
        if index + 1 < len(normalized_tokens):
            next_token, next_weight = normalized_tokens[index + 1]
            combined = token + next_token
            if (
                len(token) == 1
                and len(next_token) == 1
                and "\u4e00" <= token <= "\u9fff"
                and "\u4e00" <= next_token <= "\u9fff"
                and combined in query
            ):
                merged_tokens.append((combined, max(weight, next_weight)))
                index += 2
                continue
        merged_tokens.append((token, weight))
        index += 1

    best_weights: Dict[str, float] = {}
    for token, weight in merged_tokens:
        best_weights[token] = max(best_weights.get(token, 0.0), weight)

    keywords = sorted(
        best_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:keyword_topk]
    return result["dense_vecs"][0].tolist(), keywords


def embed(query: str) -> List[float]:
    dense_vector, _ = analyze_query(query)
    return dense_vector


def extract_keywords(
    query: str,
    topk: int = 8,
    min_weight: float = 0.05,
) -> List[Tuple[str, float]]:
    _, keywords = analyze_query(
        query,
        keyword_topk=topk,
        min_keyword_weight=min_weight,
    )
    return keywords
