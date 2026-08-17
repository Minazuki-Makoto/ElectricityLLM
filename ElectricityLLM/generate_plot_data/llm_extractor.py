from __future__ import annotations

import json
import time
from typing import Any

from AiChat.GLM_Chat import get_client

SYSTEM_PROMPT = """
你是严谨的电力技术文档结构化抽取助手。根据标题层级和正文，同时生成description和plot_data。

要求：
1. description是一个完整、准确的中文句子，只概括原文明示内容。
2. plot_data只能使用原文明示的数值，禁止计算、推测、补齐、换算或使用常识生成数据。
3. 必须抽取正文中所有对象、指标、数值和单位明确的技术参数，包括额定容量、电压、频率、调压范围、阻抗、电流、损耗、效率、温升、噪声、质量、功率、负载率、环境温度和时间序列数据。
4. 即使某项指标在当前正文中只有一个数据点，也必须建立对应数据集；不得因为数据点暂时不能形成曲线而省略。
5. 统一测试条件中的数值也必须抽取，entity填写“统一测试条件”，dimension填写“条件名称”。
6. 型号中的数字、章节号、日期、页码及无业务含义的数字不得作为数据点。
7. 同一数据集只能包含相同指标、相同value_unit和相同统计条件的数据点。
8. x保留原文横轴值；value必须是JSON数值，不能是带单位的字符串。
9. 同一指标存在多个对象或型号时，必须合并到一个数据集的points中，不要为每个数据点分别建立数据集。
10. evidence必须逐字取自正文且不超过120个汉字，只保留证明数据所需的最短片段。
11. conditions只记录该数据集所有数据点共同适用的条件，避免重复条件。
12. JSON使用紧凑格式，不添加缩进和无意义空白。
13. 仅当正文完全没有带明确含义的技术数值时，plot_data才返回空数组。
14. 只输出合法JSON，不得输出Markdown、代码围栏或解释。
15. plot_data中的每个对象只允许包含以下固定字段：topic、entity、metric、dimension、dimension_unit、value_unit、conditions、points、evidence；不得增加、改名或省略字段，不得使用“-”“对象”“设备”等动态键名。
16. entity必须填写数据所属对象的明确名称，例如“断路器”；对象名称只能作为entity的值，禁止输出{"-":"断路器"}、{"断路器":"..."}等动态字段结构。
17. points中的每个对象只允许包含label、x、value三个固定字段，不得增加其他字段；value必须是JSON数值。
18. 无法从原文确定的普通字符串字段填写空字符串，conditions填写空数组；但topic、entity、metric、dimension、value_unit和evidence应根据已抽取数据填写完整，不能用自创字段代替。

JSON格式：
{"description":"一句话摘要","plot_data":[{"topic":"数据集主题","entity":"数据所属对象","metric":"被测指标","dimension":"横轴维度","dimension_unit":"横轴单位","value_unit":"被测值单位","conditions":["适用条件"],"points":[{"label":"数据点名称","x":"原始横轴值","value":1.23}],"evidence":"支持该数据的原文"}]}

字段示例：正确写法是{"topic":"断路器额定电流","entity":"断路器","metric":"额定电流","dimension":"型号","dimension_unit":"","value_unit":"A","conditions":[],"points":[{"label":"C10B3TM100","x":"C10B3TM100","value":100}],"evidence":"C10B3TM100额定电流为100 A"}；禁止写成{"topic":"断路器额定电流","-":"断路器","metric":"额定电流"}。
""".strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == fence:
            lines.pop()
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("模型没有返回JSON对象") from original_error
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            raise ValueError("模型返回的JSON无法解析") from original_error
    if not isinstance(value, dict):
        raise ValueError("模型返回JSON的顶层必须是对象")
    return value


def extract_chunk(chunk: dict[str, Any], client: Any | None = None, model: str = "glm-4.5-air", retries: int = 3) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("retries必须大于0")
    query = f"""
文章标题：{chunk.get("title", "")}
一级标题：{chunk.get("chapter_title", "")}
二级标题：{chunk.get("section_title", "")}
三级标题：{chunk.get("third_title", "")}
四级标题：{chunk.get("fourth_title", "")}
正文：
{chunk.get("text", "")}
""".strip()
    client = client or get_client()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
                thinking={"type": "disabled"},
                temperature=0.1,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "unknown")
            if finish_reason == "length":
                raise ValueError(
                    "模型输出达到max_tokens限制，JSON被截断"
                )
            try:
                extracted = _parse_json_object(content)
            except ValueError as error:
                raise ValueError(
                    f"{error}，finish_reason={finish_reason}，"
                    f"返回字符数={len(content)}"
                ) from error
            result = chunk.copy()
            result["description"] = extracted.get("description")
            result["plot_data"] = extracted.get("plot_data")
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"chunk {chunk.get('chunk_id', '')}模型抽取失败，已重试{retries}次：{last_error}") from last_error
