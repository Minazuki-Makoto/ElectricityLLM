import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from elasticsearch.helpers import streaming_bulk

from link import link

def clear_index(es, index: str) -> bool:

    if not es.indices.exists(index=index):
        print(f"索引不存在，无需删除：{index}")

        return True

    es.indices.delete(index=index)
    print(f"旧索引已删除：{index}")

    return True

class ElasticWriter:
    VECTOR_DIMS = 1024
    VECTOR_FIELDS = ("text_embedding", "description_embedding")

    def __init__(
        self,
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        elastic_index,
        data_address,
    ):
        self.es = link(
            elasticsearch_uris,
            elasticsearch_username,
            elasticsearch_password,
        )
        self.elastic_index = elastic_index
        self.data_address = Path(data_address)

    @classmethod
    def _properties(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "chapter_title": {"type": "text"},
            "section_title": {"type": "text"},
            "third_title": {"type": "text"},
            "fourth_title": {"type": "text"},
            "text": {"type": "text"},
            "embedding_text":{
                "type": "text",
            },
            "description": {"type": "text"},
            "text_embedding": {
                "type": "dense_vector",
                "dims": cls.VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
            },
            "description_embedding": {
                "type": "dense_vector",
                "dims": cls.VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }

    def _validate_existing_mapping(self, index: str) -> None:
        mapping = self.es.indices.get_mapping(index=index)
        properties = mapping[index].get("mappings", {}).get("properties", {})
        problems = []
        for field in self.VECTOR_FIELDS:
            field_mapping = properties.get(field, {})
            if field_mapping.get("type") != "dense_vector":
                problems.append(f"{field} 不是 dense_vector")
            elif field_mapping.get("dims") != self.VECTOR_DIMS:
                problems.append(
                    f"{field} 维度为 {field_mapping.get('dims')}，"
                    f"应为 {self.VECTOR_DIMS}"
                )

        if problems:
            details = "；".join(problems)
            raise RuntimeError(
                f"索引 {index!r} 的 mapping 与当前数据不兼容：{details}。"
                "请删除旧测试索引后重建，或改用新的索引名。"
            )

    def create_index(self, index: str) -> bool:
        if self.es.indices.exists(index=index):
            self._validate_existing_mapping(index)
            return False

        self.es.indices.create(
            index=index,
            mappings={"properties": self._properties()},
        )
        return True

    @classmethod
    def _validate_chunk(cls, data: Any, position: int) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"JSON 第 {position} 项不是对象")
        if not data.get("chunk_id"):
            raise ValueError(f"JSON 第 {position} 项缺少 chunk_id")

        for field in cls.VECTOR_FIELDS:
            vector = data.get(field)
            if not isinstance(vector, list):
                raise ValueError(
                    f"chunk {data['chunk_id']} 的 {field} 不是数组"
                )
            if len(vector) != cls.VECTOR_DIMS:
                raise ValueError(
                    f"chunk {data['chunk_id']} 的 {field} 维度为 "
                    f"{len(vector)}，应为 {cls.VECTOR_DIMS}"
                )

    def _load_data(self) -> list[dict]:
        if not self.data_address.exists():
            raise FileNotFoundError(f"JSON 文件不存在：{self.data_address}")

        with self.data_address.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
        if not isinstance(json_data, list):
            raise ValueError("JSON 文件最外层必须是数组，例如 [{...}, {...}]")

        seen_ids = set()
        for position, data in enumerate(json_data, start=1):
            self._validate_chunk(data, position)
            chunk_id = data["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError(f"JSON 中存在重复 chunk_id：{chunk_id}")
            seen_ids.add(chunk_id)
        return json_data

    def _actions(self, json_data: Iterable[dict]):
        for data in json_data:
            yield {
                "_op_type": "index",
                "_index": self.elastic_index,
                "_id": data["chunk_id"],
                "_source": data,
            }

    def write_in_database(self, chunk_size: int = 50) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数")

        # 先完整验证文件，再创建索引，避免坏数据造成半成品索引。
        json_data = self._load_data()
        created = self.create_index(self.elastic_index)
        print(
            f"索引 {'已创建' if created else '已存在且 mapping 验证通过'}："
            f"{self.elastic_index}"
        )

        success_count = 0
        failures = []
        for success, info in streaming_bulk(
            self.es,
            self._actions(json_data),
            chunk_size=chunk_size,
            max_retries=3,
            initial_backoff=1,
            max_backoff=8,
            raise_on_error=False,
            raise_on_exception=False,
        ):
            if success:
                success_count += 1
                if success_count % chunk_size == 0:
                    print(f"已写入：{success_count}/{len(json_data)}")
            elif len(failures) < 5:
                failures.append(info)

        self.es.indices.refresh(index=self.elastic_index)
        if success_count != len(json_data):
            raise RuntimeError(
                f"批量写入未全部成功：成功 {success_count}/{len(json_data)}；"
                f"前几个错误：{failures}"
            )

        print(
            f"写入完成：索引={self.elastic_index}，"
            f"成功数量={success_count}"
        )


if __name__ == "__main__":
    json_dir = Path(
        r"D:\pycharmcode\ElectricityLLM\LLM-data\json_data"
    )

    index_name = "electricity-infos"
    elasticsearch_uris = "https://localhost:9200"
    elasticsearch_username = os.environ["ELASTICSEARCH_USERNAME"]
    elasticsearch_password = os.environ["ELASTICSEARCH_PASSWORD"]

    es = link(elasticsearch_uris,elasticsearch_username,elasticsearch_password)
    judge = clear_index(es, index_name)

    if judge:
        for data_file in json_dir.glob("*.json"):
            writer = ElasticWriter(
                elasticsearch_uris,
                elasticsearch_username,
                elasticsearch_password,
                index_name,
                data_file,
            )

            writer.write_in_database()

            print(f"{data_file.name}已入库完成")
