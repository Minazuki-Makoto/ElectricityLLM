import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from FlagEmbedding import FlagReranker


ROOT_PATH = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_PATH / "LLM-data" / "Models" / "reranker-model"
VERIFYING_DATA_PATH = ROOT_PATH / "LLM-data" / "verifying_data"

model = FlagReranker(
    model_name_or_path=str(MODEL_PATH),
    use_fp16=False,
)


# “问题要求”与回答中可观察到的覆盖信号。该规则只用于筛查遗漏，
# 不把关键词命中伪装成事实正确性或资料忠实度。
REQUIREMENT_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "principle": {
        "query": ("原理", "机理", "工作机制"),
        "answer": ("原理", "机理", "通过", "由于", "过程"),
    },
    "characteristics": {
        "query": ("特性", "特点", "主要性能"),
        "answer": ("特性", "特点", "具有", "表现为"),
    },
    "factors": {
        "query": ("影响因素", "考虑哪些因素", "关键因素"),
        "answer": ("因素", "影响", "取决于", "与", "有关"),
    },
    "risks": {
        "query": ("风险", "隐患", "可能故障"),
        "answer": ("风险", "隐患", "故障", "可能", "导致", "避免"),
    },
    "engineering_requirements": {
        "query": ("工程要求", "注意事项", "应强调", "重点检查", "运行维护"),
        "answer": ("应", "必须", "需要", "注意", "检查", "校核", "维护"),
    },
    "comparison": {
        "query": ("比较", "区别", "差异", "不同之处"),
        "answer": ("相比", "区别", "差异", "不同", "分别", "而"),
    },
    "logic": {
        "query": ("逻辑关系", "为什么", "原因", "因果关系"),
        "answer": ("因为", "由于", "因此", "从而", "关系", "决定", "基础"),
    },
    "standards": {
        "query": ("标准", "规程", "规范"),
        "answer": ("标准", "规程", "规范", "GB", "IEC", "IEEE"),
    },
}

REFUSAL_PATTERNS = (
    "资料不足",
    "信息不足",
    "无法从现有资料",
    "现有资料无法",
    "资料中未",
    "未提供",
    "无法确定",
    "无法分析",
)


def _validate_text_pair(query: str, answer: str) -> tuple[str, str]:
    if not isinstance(query, str) or not isinstance(answer, str):
        raise TypeError("query 和 answer 必须是字符串")
    query = query.strip()
    answer = answer.strip()
    if not query or not answer:
        raise ValueError("query 和 answer 不能为空")
    return query, answer


def calculate_relevance(
        query: str,
        answer: str,
) -> dict[str, float]:
    """计算语义相关性；该指标不代表正确性、完整性或资料忠实度。"""
    query, answer = _validate_text_pair(query, answer)
    reranker = model.model
    reranker.eval()
    device = next(reranker.parameters()).device

    inputs = model.tokenizer(
        query,
        text_pair=answer,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = reranker(
            **model_inputs,
            return_dict=True,
        ).logits

    if logits.numel() != 1:
        raise RuntimeError(
            "单个 query-answer 文本对只应产生一个 logit，"
            f"实际输出形状为 {tuple(logits.shape)}"
        )

    raw_logit = logits.reshape(()).float()
    normalized_score = torch.sigmoid(raw_logit)
    return {
        "score": float(normalized_score.cpu().item()),
        "raw_logit": float(raw_logit.cpu().item()),
    }


def calculate_related_score(
        query: str,
        answer: str,
        normalize: bool = True,
) -> float:
    """向后兼容的相关性函数；新代码应优先使用 calculate_relevance。"""
    result = calculate_relevance(query, answer)
    return result["score"] if normalize else result["raw_logit"]


def evaluate_requirement_coverage(
        query: str,
        answer: str,
) -> dict[str, Any]:
    """用可解释规则筛查问题中的明确要求是否在回答中出现。"""
    query, answer = _validate_text_pair(query, answer)
    required: list[str] = []
    covered: list[str] = []
    missing: list[str] = []
    matched_signals: dict[str, list[str]] = {}

    for name, rule in REQUIREMENT_RULES.items():
        if not any(signal in query for signal in rule["query"]):
            continue
        required.append(name)
        matches = [
            signal
            for signal in rule["answer"]
            if signal.lower() in answer.lower()
        ]
        if matches:
            covered.append(name)
            matched_signals[name] = matches
        else:
            missing.append(name)

    # 没有抽取到显式复合要求时，不把规则缺失误判成零分。
    score = (
        len(covered) / len(required)
        if required
        else None
    )
    return {
        "score": score,
        "required": required,
        "covered": covered,
        "missing": missing,
        "matched_signals": matched_signals,
        "method": "rule_based_screening",
    }


def evaluate_form_quality(
        answer: str,
        required_count: int,
        refusal_detected: bool,
) -> dict[str, Any]:
    """评价回答长度是否适中，以及末句是否有明确结束标点。"""
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", answer))

    # 正确拒答通常应比常规技术回答短，单独设置合理区间。
    if refusal_detected:
        minimum_chars = 40
        maximum_chars = 350
    else:
        minimum_chars = 120 + 80 * required_count
        maximum_chars = 600 + 150 * required_count

    if chinese_char_count < minimum_chars:
        length_score = chinese_char_count / minimum_chars
        length_status = "too_short"
    elif chinese_char_count > maximum_chars:
        length_score = maximum_chars / chinese_char_count
        length_status = "too_long"
    else:
        length_score = 1.0
        length_status = "appropriate"

    # 去掉末尾 Markdown 装饰符，再允许标点后跟右引号或右括号。
    cleaned_answer = re.sub(r"(?:\*\*|__|`)+\s*$", "", answer.strip())
    has_terminal_punctuation = bool(
        re.search(r"[。！？!?；;][”’\"'）)\]】》]*$", cleaned_answer)
    )
    completion_score = 1.0 if has_terminal_punctuation else 0.6

    # 长度占主要权重；末句疑似截断时扣除 40% 的结句子分。
    form_score = 0.75 * length_score + 0.25 * completion_score
    return {
        "score": round(form_score, 6),
        "length_score": round(length_score, 6),
        "completion_score": completion_score,
        "length_status": length_status,
        "chinese_char_count": chinese_char_count,
        "recommended_range": {
            "minimum_chinese_chars": minimum_chars,
            "maximum_chinese_chars": maximum_chars,
        },
        "has_terminal_punctuation": has_terminal_punctuation,
        "possible_truncation": not has_terminal_punctuation,
        "terminal_punctuation_rule": "回答应以 。！？?!；; 之一结句",
    }


def build_diagnostics(
        query: str,
        answer: str,
        coverage: dict[str, Any],
) -> dict[str, Any]:
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", answer))
    sentence_count = len([
        item
        for item in re.split(r"[。！？!?；;\n]+", answer)
        if item.strip()
    ])
    refusal_signals = [
        pattern
        for pattern in REFUSAL_PATTERNS
        if pattern in answer
    ]
    return {
        "answer_char_count": len(answer),
        "chinese_char_count": chinese_char_count,
        "sentence_count": sentence_count,
        "refusal_detected": bool(refusal_signals),
        "refusal_signals": refusal_signals,
    }


def calculate_overall_quality(
        relevance_score: float,
        coverage_score: float | None,
        form_score: float,
) -> dict[str, Any]:
    """合并当前可观测指标；该总分不包含事实正确性和资料忠实度。"""
    if coverage_score is None:
        weights = {
            "relevance": 0.65,
            "requirement_coverage": 0.0,
            "form_quality": 0.35,
        }
        score = (
            weights["relevance"] * relevance_score
            + weights["form_quality"] * form_score
        )
    else:
        weights = {
            "relevance": 0.35,
            "requirement_coverage": 0.45,
            "form_quality": 0.20,
        }
        score = (
            weights["relevance"] * relevance_score
            + weights["requirement_coverage"] * coverage_score
            + weights["form_quality"] * form_score
        )

    score = max(0.0, min(1.0, score))
    if score >= 0.85:
        level = "excellent"
    elif score >= 0.70:
        level = "good"
    elif score >= 0.55:
        level = "needs_improvement"
    else:
        level = "poor"

    return {
        "score": round(score, 6),
        "level": level,
        "weights": weights,
        "scope": (
            "语义相关性、明确要求覆盖度和形式质量的综合分；"
            "不包含事实正确性与资料忠实度"
        ),
    }


def evaluate_answer(
        query: str,
        answer: str,
        retrieved_contexts: list[Any] | None = None,
) -> dict[str, Any]:
    """返回多指标筛查结果，不在缺少证据时虚构总体质量分。"""
    query, answer = _validate_text_pair(query, answer)
    relevance = calculate_relevance(query, answer)
    coverage = evaluate_requirement_coverage(query, answer)
    diagnostics = build_diagnostics(query, answer, coverage)
    form_quality = evaluate_form_quality(
        answer=answer,
        required_count=len(coverage["required"]),
        refusal_detected=diagnostics["refusal_detected"],
    )
    overall_quality = calculate_overall_quality(
        relevance_score=relevance["score"],
        coverage_score=coverage["score"],
        form_score=form_quality["score"],
    )
    has_contexts = bool(retrieved_contexts)

    limitations = []
    if not has_contexts:
        limitations.append(
            "输入未包含 retrieved_contexts，无法评价资料忠实度、事实正确性和拒答是否合理"
        )
    limitations.append(
        "requirement_coverage 是规则筛查结果，需要结合人工或证据评审，不能单独视为质量总分"
    )

    return {
        "relevance": relevance,
        "requirement_coverage": coverage,
        "form_quality": form_quality,
        "diagnostics": diagnostics,
        "evidence_evaluation": {
            "available": has_contexts,
            "groundedness_score": None,
            "correctness_score": None,
            "note": (
                "已发现检索资料，但当前脚本尚未配置可靠的蕴含/事实评审模型"
                if has_contexts
                else "缺少检索资料"
            ),
        },
        "overall_quality_score": overall_quality["score"],
        "overall_quality": overall_quality,
        "limitations": limitations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 SFT 回答执行相关性与明确要求覆盖度筛查"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=VERIFYING_DATA_PATH / "verification_merged_answer.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VERIFYING_DATA_PATH / "verification_merged_evaluation.jsonl",
    )
    return parser.parse_args()


def main(
        input_path: Path | None = None,
        output_path: Path | None = None,
) -> None:
    args = parse_args() if input_path is None and output_path is None else None
    input_path = input_path or args.input
    output_path = output_path or args.output

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    scored_count = 0
    skipped_count = 0

    try:
        with (
            input_path.open("r", encoding="utf-8") as input_file,
            temporary_output_path.open("w", encoding="utf-8") as output_file,
        ):
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()
                if not line:
                    skipped_count += 1
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"输入文件第 {line_number} 行不是合法 JSON：{error.msg}"
                    ) from error
                if not isinstance(data, dict):
                    raise ValueError(
                        f"输入文件第 {line_number} 行必须是 JSON 对象"
                    )

                query = data.get("query", "")
                answer = data.get("answer", "")
                if not isinstance(query, str) or not isinstance(answer, str):
                    raise TypeError(
                        f"输入文件第 {line_number} 行的 query 和 answer 必须是字符串"
                    )
                query = query.strip()
                answer = answer.strip()
                if not query or not answer:
                    skipped_count += 1
                    continue

                contexts = data.get("retrieved_contexts")
                if contexts is not None and not isinstance(contexts, list):
                    raise TypeError(
                        f"输入文件第 {line_number} 行的 retrieved_contexts 必须是数组"
                    )

                evaluation = evaluate_answer(query, answer, contexts)
                result = {
                    # 兼容旧分析代码；该字段现在是 sigmoid 后的相关性，不是总质量分。
                    "answer_score": evaluation["relevance"]["score"],
                    "evaluation": evaluation,
                }
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                scored_count += 1

        temporary_output_path.replace(output_path)
    except Exception:
        temporary_output_path.unlink(missing_ok=True)
        raise

    print(
        f"评分完成：成功 {scored_count} 条，跳过 {skipped_count} 条，"
        f"结果已保存至 {output_path}"
    )
    print(
        "注意：overall_quality_score 仅综合相关性、明确要求覆盖度和形式质量；"
        "当前不包含事实正确性与资料忠实度。"
    )


if __name__ == "__main__":
    main()
