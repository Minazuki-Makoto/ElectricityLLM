from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from AiChat.GLM_Chat import get_client
from .config import DEFAULT_DOCUMENT_PATH, DEFAULT_MAX_CHARS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIRECTORY, DEFAULT_RETRIES
from .embedding import embed_description
from .llm_extractor import extract_chunk
from .splitter import split_docx
from .validator import validate_extracted_chunk


def _write_json_atomic(path: Path, data: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _load_completed(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"已有输出文件顶层不是数组：{path}")
    return data


def generate_plot_data(document_path: str | Path = DEFAULT_DOCUMENT_PATH, output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY, model: str = DEFAULT_MODEL, max_chars: int = DEFAULT_MAX_CHARS, retries: int = DEFAULT_RETRIES, resume: bool = False) -> Path:
    source_path = Path(document_path).expanduser().resolve()
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}.json"
    chunks = split_docx(source_path, max_chars=max_chars)
    results = _load_completed(output_path) if resume else []
    if not resume and output_path.exists():
        print(f"完整重建，将覆盖已有文件：{output_path}", flush=True)
    completed_ids = {item.get("chunk_id") for item in results if isinstance(item, dict) and item.get("chunk_id")}
    pending = [item for item in chunks if item["chunk_id"] not in completed_ids]
    if not pending:
        print(f"全部chunk已经处理：{output_path}", flush=True)
        return output_path

    client = get_client(timeout=300.0)
    metric_embedding_cache: dict[str, list[float]] = {}
    for position, chunk in enumerate(pending, start=1):
        print(f"[{position}/{len(pending)}] 抽取 {chunk['chunk_id']}", flush=True)
        enriched = extract_chunk(chunk, client=client, model=model, retries=retries)
        validate_extracted_chunk(enriched)
        enriched["description_embedding"] = embed_description(enriched["description"])
        for data in enriched["plot_data"]:
            if not isinstance(data, dict):
                raise ValueError("plot_data中的数据集格式错误")

            metric = data.get("metric")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("plot_data.metric不能为空")

            normalized_metric = metric.strip()
            metric_embedding = metric_embedding_cache.get(normalized_metric)
            if metric_embedding is None:
                metric_embedding = embed_description(normalized_metric)
                metric_embedding_cache[normalized_metric] = metric_embedding
            data["metric_embedding"] = metric_embedding

        results.append(enriched)
        _write_json_atomic(output_path, results)
    print(f"生成完成：{output_path}", flush=True)
    return output_path
