import json
import re
from pathlib import Path

SOURCE_PATH = Path(r"D:\pycharmcode\ElectricityLLM\LLM-data\temps\output-result.json")
SAVE_PATH = Path(r"D:\pycharmcode\ElectricityLLM\LLM-data\training_data\memory-training-step2.json")

PROMPT = """
你是承接问题改写器，不是问答助手。请使用当前会话最近三轮的用户问题和助手回答，补全当前问题中省略或被“它、这个、那个、继续讲”等表达指代的对象。

历史读取顺序：
1. 历史会话按时间从近到远排列，历史会话1最新。
2. 必须从历史会话1开始逐条判断；某一条已经足以唯一确定指代对象时，立即停止，不再读取或混入更早的会话。
3. 只有当前一条不足时才查看下一条；全部不足、存在多个可能对象或无法唯一确定时，判定为不能补全。

要求：
1. 能唯一补全且改写后可脱离历史独立理解时，is_able=true，new_query填写完整问题。
2. 不能唯一补全时，is_able=false，new_query必须为空字符串。
3. 保留当前问题的动作、比较关系、限制条件、数值、单位和语气，只补全缺失对象。
4. 不得增加当前问题和所采用会话中没有的事实、结论、定义或实体，不得拼接互不相关的多条会话。
5. 不回答问题；new_query不得出现“区别是、原因是、通常指、主要包括、具体来说”等答案内容。
6. 不得仅原样复制当前问题作为new_query；当前问题本身无需历史即可独立理解时，也返回false和空字符串。
7. 只输出一行合法JSON，不输出Markdown或解释。

输出格式：
{"is_able":false,"new_query":""}
""".strip()


def repair_mojibake(text: str) -> str:
    """修复常见的 GB18030 字节被按 Latin-1 保存造成的乱码。"""
    if not isinstance(text, str) or not text:
        return ""

    def repair_segment(match: re.Match[str]) -> str:
        segment = match.group(0)
        if not any(ord(char) >= 128 for char in segment):
            return segment
        try:
            repaired = segment.encode("latin-1").decode("gb18030")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return segment
        return repaired if re.search(r"[一-鿿]", repaired) else segment

    return re.sub(r"[\x00-\xff]+", repair_segment, text)


def main() -> None:
    datas = json.loads(SOURCE_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(datas, list):
        raise ValueError("output-result.json 顶层必须是数组")

    new_datas = []
    seen_inputs: set[tuple[str, str]] = set()

    for source_index, data in enumerate(datas):
        if not isinstance(data, dict) or data.get("needed_short_memory") is not True:
            continue

        original_question = repair_mojibake(
            str(data.get("original_question", "")).strip()
        )
        short_memory = repair_mojibake(
            str(data.get("short_memory", "")).strip()
        )
        if not original_question or not short_memory:
            continue

        deduplication_key = (original_question, short_memory)
        if deduplication_key in seen_inputs:
            continue
        seen_inputs.add(deduplication_key)

        new_datas.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"【当前问题】\n{original_question}\n\n"
                            f"{short_memory}"
                        ),
                    },
                ],
                # 原始结果没有保存可靠标签，人工填写后才能用于训练。
                "completion": [
                    {
                        "role": "assistant",
                        "content": "",
                    }
                ],
                "metadata": {
                    "source_index": source_index,
                    "needs_manual_label": True,
                    "label_format": '{"is_able":false,"new_query":""}',
                },
            }
        )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(
        json.dumps(new_datas, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"source_count={len(datas)}")
    print(f"cleaned_count={len(new_datas)}")
    print(f"manual_labels_required={len(new_datas)}")
    print(f"saved={SAVE_PATH}")


if __name__ == "__main__":
    main()


