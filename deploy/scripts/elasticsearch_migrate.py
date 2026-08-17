#!/usr/bin/env python3
"""Export/import Elasticsearch index settings, mappings and documents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan


DEFAULT_INDEXES = (
    "electricity-infos",
    "electricity-plot-data",
    "graph_index",
    "memory-history",
)


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def client() -> Elasticsearch:
    options = {
        "basic_auth": (
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        "verify_certs": env_bool("ELASTICSEARCH_VERIFY_CERTS"),
        "request_timeout": 120,
    }
    ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS", "").strip()
    if ca_certs:
        options["ca_certs"] = ca_certs
    return Elasticsearch(os.environ["ELASTICSEARCH_URIS"], **options)


def portable_settings(raw: dict) -> dict:
    index_settings = dict(raw.get("index", {}))
    for key in (
        "uuid", "version", "provided_name", "creation_date", "creation_date_string",
        "routing", "history", "verified_before_close", "resize", "store",
    ):
        index_settings.pop(key, None)
    return {"index": index_settings}


def export_indexes(destination: Path, indexes: list[str]) -> None:
    es = client()
    destination.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        if not es.indices.exists(index=index):
            print(f"SKIP missing index: {index}")
            continue
        metadata = {
            "settings": portable_settings(es.indices.get_settings(index=index)[index]["settings"]),
            "mappings": es.indices.get_mapping(index=index)[index]["mappings"],
        }
        (destination / f"{index}.meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        count = 0
        with (destination / f"{index}.docs.ndjson").open("w", encoding="utf-8") as stream:
            for hit in scan(es, index=index, query={"query": {"match_all": {}}}):
                stream.write(json.dumps({"_id": hit["_id"], "_source": hit["_source"]}, ensure_ascii=False))
                stream.write("\n")
                count += 1
        print(f"EXPORTED {index}: {count} documents")


def actions(path: Path, index: str):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            yield {"_op_type": "index", "_index": index, "_id": item["_id"], "_source": item["_source"]}


def import_indexes(source: Path, indexes: list[str]) -> None:
    es = client()
    for index in indexes:
        meta_path = source / f"{index}.meta.json"
        docs_path = source / f"{index}.docs.ndjson"
        if not meta_path.exists() or not docs_path.exists():
            print(f"SKIP missing export files: {index}")
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not es.indices.exists(index=index):
            es.indices.create(index=index, settings=metadata["settings"], mappings=metadata["mappings"])
        success, errors = bulk(es, actions(docs_path, index), chunk_size=200, request_timeout=120)
        es.indices.refresh(index=index)
        print(f"IMPORTED {index}: {success} documents, errors={len(errors)}")


def verify(indexes: list[str]) -> None:
    es = client()
    for index in indexes:
        if es.indices.exists(index=index):
            print(f"{index}\t{es.count(index=index)['count']}")
        else:
            print(f"{index}\tMISSING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "import", "verify"))
    parser.add_argument("--directory", type=Path, default=Path("./backup/elasticsearch"))
    parser.add_argument("--indexes", nargs="+", default=list(DEFAULT_INDEXES))
    args = parser.parse_args()
    if args.action == "export":
        export_indexes(args.directory, args.indexes)
    elif args.action == "import":
        import_indexes(args.directory, args.indexes)
    else:
        verify(args.indexes)


if __name__ == "__main__":
    main()
