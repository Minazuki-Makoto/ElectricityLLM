import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = ROOT / "LLM-data" / "Models" / "small-model"
MERGED_MODEL = ROOT / "LLM-data" / "Models" / "new-small-model-2"

DATASET = ROOT / "LLM-data" / "verifying_data" / "verification_merged.jsonl"
OUTPUT_DIR = ROOT / "LLM-data" / "verifying_data"

SYSTEM_PROMPT = '''你是记忆路由器，只分析用户输入，不回答问题。只输出一行 JSON，不得输出解释、Markdown或第二个 JSON。

判定规则：
1. 当前输入依赖上一轮对象（如“继续讲”“再详细说”“它怎么样”且本句没有明确对象）时，short=true；本句已给出对象时为 false。
2. 回答必须使用用户既有专业、职业、兴趣或长期绘图偏好时，long=true，否则为 false。
3. 仅提取用户以本人身份明确表达、跨会话稳定的画像更新；示例、假设、第三方、临时要求不得保存。
4. 允许字段仅为 user_true_name、major、profession、interests、prefer_plot_style。
5. 明确提供或纠正使用 update；明确要求忘记某字段使用 delete，delete 的 value 为 null。

固定格式：
{"short":false,"long":false,"operations":[]}
更新示例：
{"short":false,"long":false,"operations":[{"operation":"update","field":"major","value":"电气工程"}]}'''


def parse_args():
    parser = argparse.ArgumentParser(description="对比验证原始小模型和融合后小模型")
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--merged-model", type=Path, default=MERGED_MODEL)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_dataset(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("query"), str) or not row["query"].strip():
                raise ValueError(f"第 {line_number} 行 query 异常")
            expected = row.get("expected")
            if not isinstance(expected, dict) or set(expected) != {"short", "long", "operations"}:
                raise ValueError(f"第 {line_number} 行 expected 异常")
            rows.append(row)
    if not rows:
        raise ValueError("验证集为空")
    return rows


def load_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU")
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def parse_json_object(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return value


def predict(model, tokenizer, query: str, max_new_tokens: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def binary_metrics(expected, predicted, field):
    tp = sum(e[field] is True and p.get(field) is True for e, p in zip(expected, predicted))
    fp = sum(e[field] is False and p.get(field) is True for e, p in zip(expected, predicted))
    fn = sum(e[field] is True and p.get(field) is not True for e, p in zip(expected, predicted))
    accuracy = sum(p.get(field) is e[field] for e, p in zip(expected, predicted)) / len(expected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def verify_model(model_name, model_path, rows, results_path, summary_path, max_new_tokens):
    print(f"\n开始验证 {model_name}：{model_path}")
    model, tokenizer = load_model(model_path)
    records, valid_predictions = [], []
    expected_rows = [row["expected"] for row in rows]
    valid_json = exact = operations_exact = 0

    for index, row in enumerate(rows, 1):
        raw = predict(model, tokenizer, row["query"], max_new_tokens)
        error = None
        try:
            parsed = parse_json_object(raw)
            valid_json += 1
        except (json.JSONDecodeError, ValueError) as exc:
            parsed, error = {}, str(exc)
        valid_predictions.append(parsed)
        is_exact = parsed == row["expected"]
        exact += is_exact
        operation_match = parsed.get("operations") == row["expected"]["operations"]
        operations_exact += operation_match
        records.append({
            "model": model_name,
            "id": row.get("id", f"verify-{index:03d}"),
            "query": row["query"],
            "expected": row["expected"],
            "raw_output": raw,
            "parsed_output": parsed if parsed else None,
            "valid_json": error is None,
            "exact_match": is_exact,
            "operations_exact": operation_match,
            "error": error,
        })
        print(f"[{model_name} {index:02d}/{len(rows)}] exact={is_exact} query={row['query']}")

    count = len(rows)
    summary = {
        "model": model_name,
        "model_path": str(model_path),
        "samples": count,
        "json_valid_rate": valid_json / count,
        "exact_match_rate": exact / count,
        "operations_exact_rate": operations_exact / count,
        "short": binary_metrics(expected_rows, valid_predictions, "short"),
        "long": binary_metrics(expected_rows, valid_predictions, "long"),
    }
    results_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"逐条结果：{results_path}")
    print(f"汇总结果：{summary_path}")
    del model, tokenizer
    torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    rows = load_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [
        ("base", args.base_model),
        ("merged", args.merged_model),
    ]
    summaries = []
    for model_name, model_path in models:
        summaries.append(verify_model(
            model_name,
            model_path,
            rows,
            args.output_dir / f"memory_routing_{model_name}_results.jsonl",
            args.output_dir / f"memory_routing_{model_name}_summary.json",
            args.max_new_tokens,
        ))

    comparison_path = args.output_dir / "memory_routing_model_comparison.json"
    comparison_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型对比汇总：{comparison_path}")


if __name__ == "__main__":
    main()
