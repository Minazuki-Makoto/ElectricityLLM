from __future__ import annotations

import math
from typing import Any


def validate_extracted_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    description = chunk.get("description")
    plot_data = chunk.get("plot_data")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description必须是非空字符串")
    if not isinstance(plot_data, list):
        raise ValueError("plot_data必须是数组")
    for dataset_index, dataset in enumerate(plot_data):
        if not isinstance(dataset, dict):
            raise ValueError(f"plot_data[{dataset_index}]必须是对象")

        # 模型在字段没有明确内容时可能返回 JSON null。当前数据结构允许字符串
        # 字段为空，因此统一转换为空字符串；数字、列表、字典等错误类型仍拒绝。
        string_fields = (
            "topic",
            "entity",
            "metric",
            "dimension",
            "dimension_unit",
            "value_unit",
            "evidence",
        )
        for field in string_fields:
            if dataset.get(field) is None:
                dataset[field] = ""

        for field in string_fields:
            if not isinstance(dataset.get(field), str):
                raise ValueError(f"plot_data[{dataset_index}].{field}必须是字符串")
        conditions = dataset.get("conditions")
        if conditions is None:
            conditions = []
            dataset["conditions"] = conditions
        elif isinstance(conditions, str):
            conditions = [conditions.strip()] if conditions.strip() else []
            dataset["conditions"] = conditions
        if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
            raise ValueError(f"plot_data[{dataset_index}].conditions必须是字符串数组")
        points = dataset.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"plot_data[{dataset_index}].points必须是非空数组")
        for point_index, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError(f"plot_data[{dataset_index}].points[{point_index}]必须是对象")
            value = point.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"plot_data[{dataset_index}].points[{point_index}].value必须是数值")
            if not math.isfinite(float(value)):
                raise ValueError("plot_data中的value不能是NaN或无穷大")
    chunk["description"] = description.strip()
    return chunk
