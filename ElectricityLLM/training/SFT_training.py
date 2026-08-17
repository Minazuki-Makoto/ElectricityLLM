"""Qwen3-4B Electricity SFT with QLoRA on a Linux NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ROLES = ["system", "user", "assistant"]


PROJECT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_DIR / "LLM-data" / "Models" / "LLM" / "merged" / "Qwen3-4B-Instruct-electricity-sft-v1"
DEFAULT_DATA = PROJECT_DIR / "LLM-data" / "training_data" / "output-data-1000.json"
DEFAULT_OUTPUT = PROJECT_DIR / "LLM-data" / "Models" / "LLM" / "adapters" / "qwen3-4b-electricity-sft-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ElectricityLLM QLoRA SFT")

    #不带--表示位置参数
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model directory or Hugging Face model id")
    parser.add_argument("--data", action="append", help="SFT JSON file; repeat for multiple files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_samples(paths: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped_insufficient: list[str] = []
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
            if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
                raise ValueError(f"{path} 第 {index} 条含空消息")
            sample_id = str(metadata.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"sample_id 缺失或重复：{sample_id!r}")
            if metadata.get("dataset_split") not in {"train", "validation", "test"}:
                raise ValueError(f"{sample_id} 的 dataset_split 不合法")
            seen_ids.add(sample_id)
            if metadata.get("context_sufficiency") is not True or not metadata.get("chunk_ids"):
                skipped_insufficient.append(sample_id)
                continue
            samples.append(sample)
    if skipped_insufficient:
        preview = ", ".join(skipped_insufficient[:10])
        suffix = " ..." if len(skipped_insufficient) > 10 else ""
        print(
            f"warning: skipped {len(skipped_insufficient)} insufficient/empty-retrieval samples: "
            f"{preview}{suffix}"
        )
    return samples


def print_summary(samples: list[dict[str, Any]]) -> None:
    split = Counter(sample["metadata"]["dataset_split"] for sample in samples)
    difficulty = Counter(sample["metadata"].get("difficulty") for sample in samples)
    question_type = Counter(sample["metadata"].get("question_type") for sample in samples)
    print(f"samples={len(samples)} split={dict(split)}")
    print(f"difficulty={dict(difficulty)}")
    print(f"question_type={dict(question_type)}")
    if not split["train"] or not split["validation"]:
        raise ValueError("训练集和验证集都必须至少包含一条样本")


def to_prompt_completion(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample["messages"]
    # Conversational prompt-completion format makes TRL mask system/user tokens
    # and compute loss on the assistant completion only.
    return {
        "prompt": messages[:2],
        "completion": [messages[2]],
        "sample_id": sample["metadata"]["sample_id"],
    }


def main() -> None:
    args = parse_args()
    if not args.data:
        args.data = [str(DEFAULT_DATA)]
    samples = read_samples(args.data)
    print_summary(samples)
    if args.validate_only:
        print("validation_ok")
        return

    # Heavy ML imports happen only for real training, so local data validation
    # works even when the Windows machine does not have a CUDA PyTorch build.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU；请在华为云 NVIDIA GPU 实例上运行训练")

    set_seed(args.seed)
    random.seed(args.seed)
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    capability_major, capability_minor = compute_capability

    # Native BF16 requires an Ampere-generation GPU (compute capability 8.0)
    # or newer. PyTorch may report BF16 as available based on the CUDA build
    # alone, so the hardware capability must also be checked. Tesla P100 has
    # compute capability 6.0 and therefore uses FP16.

    supports_bf16 = (
        capability_major >= 8
        and torch.cuda.is_bf16_supported()
    )

    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(
        f"gpu={gpu_name} "
        f"capability={capability_major}.{capability_minor} "
        f"compute_dtype={compute_dtype}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
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

    print(f"active_adapters={model.active_adapters}")

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

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise RuntimeError("LoRA 未注入任何可训练参数")
    print(f"trainable_parameters={trainable_parameters}")

    train_rows = [to_prompt_completion(x) for x in samples if x["metadata"]["dataset_split"] == "train"]
    eval_rows = [to_prompt_completion(x) for x in samples if x["metadata"]["dataset_split"] == "validation"]
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Trainer construction must not replace, freeze, or disable the adapter.
    trainer.model.enable_adapter_layers()
    trainer.model.set_adapter("default")
    trainer.model.train()
    trainer_trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    print(f"trainer_trainable_parameters={trainer_trainable_parameters}")
    if trainer_trainable_parameters == 0:
        raise RuntimeError("Trainer 创建后 LoRA 参数全部被冻结")

    def check_labels(dataloader: Any, name: str) -> None:
        """Fail before training if truncation removes every assistant label."""
        zero_label_batches = 0
        minimum_valid_labels: int | None = None
        total_batches = 0
        for batch in dataloader:
            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError(f"{name} batch 缺少 labels")
            valid_labels = int((labels != -100).sum().item())
            total_batches += 1
            minimum_valid_labels = (
                valid_labels
                if minimum_valid_labels is None
                else min(minimum_valid_labels, valid_labels)
            )
            if valid_labels == 0:
                zero_label_batches += 1
        print(
            f"{name}_label_check: batches={total_batches}, "
            f"zero_label_batches={zero_label_batches}, "
            f"min_valid_labels={minimum_valid_labels}"
        )
        if zero_label_batches:
            raise RuntimeError(
                f"{name} 中有 {zero_label_batches} 个 batch 的 assistant 标签被全部截断；"
                "请增大 --max-length 或缩短检索上下文"
            )

    check_labels(trainer.get_train_dataloader(), "train")
    check_labels(trainer.get_eval_dataloader(), "validation")

    # Run one real backward pass before paid training time is consumed. This
    # catches broken checkpointing, missing LoRA gradients, and FP16 NaN/Inf.
    preflight_batch = next(iter(trainer.get_train_dataloader()))
    preflight_batch = {
        key: value.to(trainer.accelerator.device)
        for key, value in preflight_batch.items()
        if hasattr(value, "to")
    }
    trainer.model.zero_grad(set_to_none=True)
    # Use the same gradient/autocast context as Trainer and explicitly enable
    # autograd so this check cannot inherit an evaluation/no-grad context.
    with torch.enable_grad(), trainer.compute_loss_context_manager():
        preflight_outputs = trainer.model(**preflight_batch)
        preflight_loss = preflight_outputs.loss
    logits_require_grad = bool(preflight_outputs.logits.requires_grad)
    print(f"preflight_loss_requires_grad={preflight_loss.requires_grad}")
    print(f"preflight_logits_require_grad={logits_require_grad}")
    if not preflight_loss.requires_grad or not logits_require_grad:
        raise RuntimeError(
            "LoRA 层虽然已启用，但没有参与模型前向计算；"
            "请检查 PEFT 与模型实现的兼容性"
        )
    if not torch.isfinite(preflight_loss):
        raise RuntimeError(f"预检 loss 为 NaN/Inf：{preflight_loss.item()}")
    trainer.accelerator.backward(preflight_loss)

    gradient_tensor_count = 0
    gradient_norm_squared = 0.0
    for parameter in trainer.model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient_tensor_count += 1
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise RuntimeError("预检发现 LoRA 梯度包含 NaN/Inf")
        gradient_norm_squared += gradient.pow(2).sum().item()

    preflight_gradient_norm = gradient_norm_squared ** 0.5
    print(f"preflight_loss={preflight_loss.item():.6f}")
    print(f"preflight_gradient_tensors={gradient_tensor_count}")
    print(f"preflight_gradient_norm={preflight_gradient_norm:.6f}")
    if gradient_tensor_count == 0 or preflight_gradient_norm == 0.0:
        raise RuntimeError("LoRA 参数没有获得有效梯度，已在正式训练前停止")

    trainer.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adapter_saved={output_dir / 'final_adapter'}")
    print(f"eval_metrics={metrics}")


if __name__ == "__main__":
    main()