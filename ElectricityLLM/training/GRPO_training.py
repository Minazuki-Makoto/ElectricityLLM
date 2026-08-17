"""Qwen3-4B Electricity GRPO with QLoRA on a Linux NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from trl import GRPOConfig, GRPOTrainer

EXPECTED_ROLES = ["system", "user", "assistant"]

PROJECT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

DEFAULT_MODEL = (
    PROJECT_DIR
    / "LLM-data"
    / "Models"
    / "LLM"
    / "merged"
    / "Qwen3-4B-Instruct-electricity-sft-v1"
)

DEFAULT_DATA = PROJECT_DIR / "LLM-data" / "training_data" / "output-data.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "LLM-data"
    / "Models"
    / "LLM"
    / "adapters"
    / "qwen3-4b-electricity-grpo-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ElectricityLLM QLoRA GRPO")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", action="append", help="GRPO source JSON; repeat for multiple files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_samples(paths: list[str]) -> list[dict[str, Any]]:
    """Read and validate the existing SFT source JSON without discarding insufficient samples."""
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"训练数据不存在：{path}")

        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError(f"训练数据必须是 JSON 数组：{path}")

        for index, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(f"{path} 第 {index} 条不是对象")

            messages = sample.get("messages")
            metadata = sample.get("metadata")
            if not isinstance(messages, list) or not isinstance(metadata, dict):
                raise ValueError(f"{path} 第 {index} 条缺少 messages 或 metadata")

            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if roles != EXPECTED_ROLES:
                raise ValueError(f"{path} 第 {index} 条角色顺序错误：{roles}")
            if any(
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            ):
                raise ValueError(f"{path} 第 {index} 条含空消息")

            sample_id = str(metadata.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"sample_id 缺失或重复：{sample_id!r}")

            dataset_split = metadata.get("dataset_split")
            if dataset_split not in {"train", "validation", "test"}:
                raise ValueError(f"{sample_id} 的 dataset_split 不合法")

            context_sufficiency = metadata.get("context_sufficiency")
            if not isinstance(context_sufficiency, bool):
                raise ValueError(f"{sample_id} 的 context_sufficiency 必须是布尔值")

            seen_ids.add(sample_id)
            samples.append(sample)

    return samples


def parse_user_content(content: str, sample_id: str) -> tuple[str, list[dict[str, str]]]:
    """Extract the raw question and source blocks from the existing SFT user message."""
    question_marker = "【问题】"
    if question_marker not in content:
        question_marker = "【用户问题】"
    if question_marker not in content:
        raise ValueError(f"{sample_id} 的 user 消息缺少问题标记")

    context_part, query = content.rsplit(question_marker, maxsplit=1)
    query = query.strip()
    if not query:
        raise ValueError(f"{sample_id} 的原始问题为空")

    context_part = context_part.replace("【检索资料】", "", 1).strip()
    blocks = re.findall(
        r"\[source_\d+\]\s*(.*?)(?=\n\s*\[source_\d+\]|\Z)",
        context_part,
        flags=re.DOTALL,
    )

    contexts: list[dict[str, str]] = []
    for block in blocks:
        title_match = re.search(r"^document_title:\s*(.*?)\s*$", block, flags=re.MULTILINE)
        chunk_match = re.search(r"^chunk_id:\s*(.*?)\s*$", block, flags=re.MULTILINE)
        text = re.sub(r"^(?:document_title|chunk_id):.*?\n", "", block, flags=re.MULTILINE).strip()
        if not text:
            continue
        contexts.append(
            {
                "document_title": title_match.group(1).strip() if title_match else "",
                "chunk_id": chunk_match.group(1).strip() if chunk_match else "",
                "text": text,
            }
        )

    if not contexts and context_part:
        contexts.append({"document_title": "", "chunk_id": "", "text": context_part})

    return query, contexts


def to_grpo_row(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample["messages"]
    metadata = sample["metadata"]
    sample_id = str(metadata["sample_id"])
    query, retrieved_contexts = parse_user_content(messages[1]["content"], sample_id)

    return {
        "prompt": messages[:2],
        "query": query,
        "retrieved_contexts": retrieved_contexts,
        "context_sufficient": bool(metadata["context_sufficiency"]),
        "sample_id": sample_id,
        "dataset_split": metadata["dataset_split"],
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    split_counts = Counter(row["dataset_split"] for row in rows)
    sufficient_counts = Counter(row["context_sufficient"] for row in rows)
    context_counts = [len(row["retrieved_contexts"]) for row in rows]
    print(f"samples={len(rows)} split={dict(split_counts)}")
    print(f"context_sufficient={dict(sufficient_counts)}")
    print(
        "retrieved_contexts="
        f"min={min(context_counts, default=0)} "
        f"max={max(context_counts, default=0)}"
    )
    if not split_counts["train"] or not split_counts["validation"]:
        raise ValueError("训练集和验证集都必须至少包含一条样本")


def extract_answers(completions: list[Any]) -> list[str]:
    """Support TRL conversational completions and plain-text completions."""
    answers: list[str] = []
    for completion in completions:
        if (
            isinstance(completion, list)
            and completion
            and isinstance(completion[0], dict)
        ):
            answer = completion[0].get("content", "")
        elif isinstance(completion, str):
            answer = completion
        else:
            answer = ""
        answers.append(answer if isinstance(answer, str) else "")
    return answers


def _check_aligned(name: str, answers: list[str], *columns: list[Any]) -> None:
    expected = len(answers)
    if any(len(column) != expected for column in columns):
        lengths = [expected, *(len(column) for column in columns)]
        raise ValueError(f"{name} 奖励函数输入数量不一致：{lengths}")


def main() -> None:
    args = parse_args()
    if not args.data:
        args.data = [str(DEFAULT_DATA)]

    samples = read_samples(args.data)
    rows = [to_grpo_row(sample) for sample in samples]
    print_summary(rows)

    if args.validate_only:
        print("validation_ok")
        return

    # Match SFT_training.py: import heavy ML dependencies only for real cloud training.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


    from rules import (
        basic_quality_reward,
        compute_related_score,
        context_grounding_reward,
        meta_phrase_penalty,
        query_copy_penalty,
        refusal_consistency_reward,
        repetition_penalty,
        sentence_completion_reward,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU；请在华为云 NVIDIA GPU 实例上运行训练")

    set_seed(args.seed)
    random.seed(args.seed)
    gpu_name = torch.cuda.get_device_name(0)
    capability_major, capability_minor = torch.cuda.get_device_capability(0)
    supports_bf16 = capability_major >= 8 and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(
        f"gpu={gpu_name} "
        f"capability={capability_major}.{capability_minor} "
        f"compute_dtype={compute_dtype}"
    )

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    checkpointing_kwargs = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.enable_adapter_layers()
    model.set_adapter("default")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=checkpointing_kwargs,
    )

    adapter_layer_count = 0
    disabled_adapter_layer_count = 0
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            adapter_layer_count += 1
            if module.disable_adapters:
                disabled_adapter_layer_count += 1
    print(
        f"adapter_layers={adapter_layer_count} "
        f"disabled_adapter_layers={disabled_adapter_layer_count}"
    )
    if adapter_layer_count == 0 or disabled_adapter_layer_count:
        raise RuntimeError("LoRA 层未注入或仍处于禁用状态")
    model.print_trainable_parameters()

    train_rows = [row for row in rows if row["dataset_split"] == "train"]
    eval_rows = [row for row in rows if row["dataset_split"] == "validation"]
    for row in train_rows + eval_rows:
        row.pop("dataset_split", None)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    def basic_reward_func(completions, **kwargs) -> list[float]:
        return [float(basic_quality_reward(answer)) for answer in extract_answers(completions)]

    def meta_reward_func(completions, **kwargs) -> list[float]:
        return [float(meta_phrase_penalty(answer)) for answer in extract_answers(completions)]

    def repetition_reward_func(completions, **kwargs) -> list[float]:
        return [float(repetition_penalty(answer)) for answer in extract_answers(completions)]

    def completion_reward_func(completions, **kwargs) -> list[float]:
        return [float(sentence_completion_reward(answer)) for answer in extract_answers(completions)]

    def refusal_reward_func(completions, context_sufficient, **kwargs) -> list[float]:
        answers = extract_answers(completions)
        _check_aligned("refusal", answers, context_sufficient)
        return [
            float(refusal_consistency_reward(answer, bool(sufficient)))
            for answer, sufficient in zip(answers, context_sufficient)
        ]

    def query_relevance_reward_func(completions, query, **kwargs) -> list[float]:
        answers = extract_answers(completions)
        _check_aligned("query_relevance", answers, query)
        return [
            float(compute_related_score(current_query, answer))
            for answer, current_query in zip(answers, query)
        ]

    def query_copy_reward_func(completions, query, **kwargs) -> list[float]:
        answers = extract_answers(completions)
        _check_aligned("query_copy", answers, query)
        return [
            float(query_copy_penalty(current_query, answer))
            for answer, current_query in zip(answers, query)
        ]

    def grounding_reward_func(
        completions,
        retrieved_contexts,
        context_sufficient,
        **kwargs,
    ) -> list[float]:
        answers = extract_answers(completions)
        _check_aligned("grounding", answers, retrieved_contexts, context_sufficient)
        return [
            float(context_grounding_reward(answer, contexts, bool(sufficient)))
            for answer, contexts, sufficient in zip(
                answers,
                retrieved_contexts,
                context_sufficient,
            )
        ]

    reward_funcs = [
        basic_reward_func,
        meta_reward_func,
        repetition_reward_func,
        completion_reward_func,
        refusal_reward_func,
        query_relevance_reward_func,
        query_copy_reward_func,
        grounding_reward_func,
    ]
    reward_weights = [0.10, 0.10, 0.10, 0.10, 0.20, 0.15, 0.10, 0.15]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        beta=args.beta,
        num_iterations=1,
        epsilon=0.2,
        loss_type="dapo",
        scale_rewards="group",
        reward_weights=reward_weights,
        remove_unused_columns=False,
        mask_truncated_completions=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    train_first_run_check(trainer)
    trainer.model.enable_adapter_layers()
    trainer.model.set_adapter("default")
    trainer.model.train()
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    print(f"trainer_trainable_parameters={trainable_parameters}")
    if trainable_parameters == 0:
        raise RuntimeError("Trainer 创建后 LoRA 参数全部被冻结")

    first_row = train_dataset[0]
    print(f"train_columns={train_dataset.column_names}")
    print(
        f"preflight_sample={first_row['sample_id']} "
        f"query_length={len(first_row['query'])} "
        f"contexts={len(first_row['retrieved_contexts'])}"
    )


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"adapter_saved={final_dir}")
    print(f"eval_metrics={metrics}")

def train_first_run_check(
        trainer:GRPOTrainer
):
    trainer.model.train()
    trainer.model.zero_grad(
        set_to_none=True
    )

    primary_batch = next(iter(trainer.get_train_dataloader()))

    if isinstance(primary_batch,dict):
        keys = primary_batch.keys()
        print("训练集样本的key如下：")
        print(keys)


    elif isinstance(primary_batch,list) and primary_batch:

        if isinstance(primary_batch[0],dict):
            print("训练集样本的key如下：")
            print(primary_batch[0].keys())

        else:
            raise ValueError("数据格式不符合要求")
    else:
        raise ValueError("数据格式不符合要求")
    primary_input = trainer._prepare_inputs(
        primary_batch
    )
    with (
        torch.enable_grad() ,
        trainer.compute_loss_context_manager()
    ):

        loss = trainer.compute_loss(
            model = trainer.model,
            inputs = primary_input
        )

        if loss.ndim >0:
            loss = loss.mean()

        print(f"peft_loss_requiredgrad = {loss.requires_grad}")
        print(f"peft_loss = {loss.detach().item()}")
        if not loss.requires_grad:
            raise ValueError("loss没有梯度，初始化检查失败")

        if not torch.isfinite(loss).item():
            raise ValueError("loss过大，运算出现问题")

    trainer.accelerator.backward(loss)

    gradient_count = 0
    gradient_pow_value = 0.0
    trainable_parameters_without_grad = []

    for name,parameter in trainer.model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            trainable_parameters_without_grad.append(
                name
            )
            continue

        gradient_count += 1

        if not torch.isfinite(parameter.grad).all().item() :
            raise ValueError("梯度出现Nan,运行停止")

        gradient_pow_value += (
            parameter.grad.pow(2).sum().item()
        )

    if gradient_count == 0 or gradient_pow_value ==0:
        raise ValueError("所有参数梯度为0，回传失败，停止运行")

    trainer.model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("==============================\n\n预检通过")



if __name__ == "__main__":
    main()
