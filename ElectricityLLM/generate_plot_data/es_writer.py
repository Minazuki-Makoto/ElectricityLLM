from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from .config import DEFAULT_DOCUMENT_PATH, DEFAULT_OUTPUT_DIRECTORY


DEFAULT_INDEX = "electricity-plot-data"


def create_client(
    uri: str,
    username: str,
    password: str,
) -> Elasticsearch:
    return Elasticsearch(
        uri,
        basic_auth=(username, password),
        request_timeout=30,
        verify_certs=False,
        ssl_show_warn=False,
    )


def load_plot_json(json_path: str | Path) -> list[dict[str, Any]]:
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"绘图JSON不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError("绘图JSON顶层必须是非空数组")
    return data


def validate_documents(documents: list[dict[str, Any]]) -> int:
    seen_ids: set[str] = set()
    vector_dims: int | None = None

    for position, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise ValueError(f"第{position}条记录不是对象")

        chunk_id = document.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"第{position}条记录缺少chunk_id")
        if chunk_id in seen_ids:
            raise ValueError(f"存在重复chunk_id：{chunk_id}")
        seen_ids.add(chunk_id)

        for field in ("text", "description"):
            value = document.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"chunk {chunk_id}的{field}为空")

        vector = document.get("description_embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"chunk {chunk_id}的description_embedding为空")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in vector
        ):
            raise ValueError(f"chunk {chunk_id}的embedding包含非数值")
        if vector_dims is None:
            vector_dims = len(vector)
        elif len(vector) != vector_dims:
            raise ValueError(
                f"chunk {chunk_id}的embedding维度为{len(vector)}，"
                f"预期为{vector_dims}"
            )

        plot_data = document.get("plot_data")
        if not isinstance(plot_data, list):
            raise ValueError(f"chunk {chunk_id}的plot_data不是数组")

    if vector_dims is None:
        raise ValueError("无法确定embedding维度")
    return vector_dims


def build_mapping(vector_dims: int) -> dict[str, Any]:
    if vector_dims <= 0:
        raise ValueError("vector_dims必须大于0")
    return {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "source": {"type": "keyword", "index": False},
            "title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "chapter_title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "section_title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "third_title": {"type": "text"},
            "fourth_title": {"type": "text"},
            "text": {"type": "text"},
            "description": {"type": "text"},
            "description_embedding": {
                "type": "dense_vector",
                "dims": vector_dims,
                "index": True,
                "similarity": "cosine",
            },
            "plot_data": {
                "type": "nested",
                "properties": {
                    "topic": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "entity": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "metric": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "dimension": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "metric_embedding": {
                        "type": "dense_vector",
                        "dims": vector_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "dimension_unit": {"type": "keyword"},
                    "value_unit": {"type": "keyword"},
                    "conditions": {"type": "keyword"},
                    "evidence": {"type": "text", "index": False},
                    "points": {
                        "type": "nested",
                        "properties": {
                            "label": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}},
                            },
                            "x": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}},
                            },
                            "value": {"type": "double"},
                        },
                    },
                },
            },
        },
    }


def prepare_index(
    client: Elasticsearch,
    index: str,
    vector_dims: int,
    recreate: bool = False,
) -> bool:
    exists = bool(client.indices.exists(index=index))
    if exists and recreate:
        client.indices.delete(index=index)
        exists = False
    if not exists:
        client.indices.create(
            index=index,
            mappings=build_mapping(vector_dims),
        )
        return True

    mapping = client.indices.get_mapping(index=index)
    properties = mapping[index].get("mappings", {}).get("properties", {})
    vector_mapping = properties.get("description_embedding", {})
    if vector_mapping.get("type") != "dense_vector":
        raise RuntimeError("已有索引的description_embedding不是dense_vector")
    if vector_mapping.get("dims") != vector_dims:
        raise RuntimeError(
            f"已有索引向量维度为{vector_mapping.get('dims')}，"
            f"当前数据维度为{vector_dims}"
        )
    if properties.get("plot_data", {}).get("type") != "nested":
        raise RuntimeError("已有索引的plot_data不是nested类型")
    return False


def _actions(
    documents: Iterable[dict[str, Any]],
    index: str,
) -> Iterable[dict[str, Any]]:
    for document in documents:
        yield {
            "_op_type": "index",
            "_index": index,
            "_id": document["chunk_id"],
            "_source": document,
        }


def write_plot_data(
    client: Elasticsearch,
    index: str,
    documents: list[dict[str, Any]],
    chunk_size: int = 50,
    recreate: bool = False,
) -> dict[str, Any]:
    if not index or not index.strip():
        raise ValueError("index不能为空")
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    vector_dims = validate_documents(documents)
    created = prepare_index(client, index, vector_dims, recreate=recreate)

    success_count = 0
    failures: list[dict[str, Any]] = []
    for success, information in streaming_bulk(
        client,
        _actions(documents, index),
        chunk_size=chunk_size,
        max_retries=3,
        initial_backoff=1,
        max_backoff=8,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if success:
            success_count += 1
        elif len(failures) < 10:
            failures.append(information)

    client.indices.refresh(index=index)
    if success_count != len(documents):
        raise RuntimeError(
            f"ES写入未全部成功：成功{success_count}/{len(documents)}，"
            f"前几个错误：{failures}"
        )
    return {
        "index": index,
        "created": created,
        "document_count": success_count,
        "vector_dims": vector_dims,
    }


def import_json_to_es(
    json_path: str | Path,
    elasticsearch_uri: str,
    elasticsearch_username: str,
    elasticsearch_password: str,
    index: str = DEFAULT_INDEX,
    chunk_size: int = 50,
    recreate: bool = False,
) -> dict[str, Any]:
    documents = load_plot_json(json_path)
    client = create_client(
        elasticsearch_uri,
        elasticsearch_username,
        elasticsearch_password,
    )
    try:
        if not client.ping():
            raise ConnectionError("无法连接Elasticsearch")
        return write_plot_data(
            client,
            index,
            documents,
            chunk_size=chunk_size,
            recreate=recreate,
        )
    finally:
        client.close()


def main() -> None:
    default_json = (
        DEFAULT_OUTPUT_DIRECTORY
        / f"{DEFAULT_DOCUMENT_PATH.stem}.json"
    )

    parser = argparse.ArgumentParser(
        description="将绘图JSON写入Elasticsearch"
    )
    parser.add_argument("--json", default=str(default_json))
    parser.add_argument(
        "--uri",
        default=os.getenv("ELASTICSEARCH_URI", "https://localhost:9200"),
    )
    parser.add_argument(
        "--username",
        default=os.getenv("ELASTICSEARCH_USERNAME"),
    )
    parser.add_argument(
        "--password",
        default=os.getenv("ELASTICSEARCH_PASSWORD"),
    )
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除同名索引后重建",
    )
    arguments = parser.parse_args()
    if not arguments.username or not arguments.password:
        parser.error(
            "必须通过参数或环境变量提供Elasticsearch用户名和密码"
        )

    result = import_json_to_es(
        json_path=arguments.json,
        elasticsearch_uri=arguments.uri,
        elasticsearch_username=arguments.username,
        elasticsearch_password=arguments.password,
        index=arguments.index,
        chunk_size=arguments.chunk_size,
        recreate=arguments.recreate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
