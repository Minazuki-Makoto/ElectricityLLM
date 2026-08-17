import json
from collections import Counter
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from trl import SFTConfig,SFTTrainer
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from peft.tuners.tuners_utils import BaseTunerLayer

from datasets import Dataset
import torch
from pathlib import Path

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SFT_TRAINING_DATA = PROJECT_ROOT / 'LLM-data' / 'training_data'/"memory-training-step2.json"

SMALL_MODEL_PATH = PROJECT_ROOT / 'LLM-data' / 'Models' / "new-small-model-2"

OUTPUT_DIR = PROJECT_ROOT / 'LLM-data'/"Models"/"LLM"/"adapters"/"qwen-0.6B_sft4_adapters"

def load_data():

    with open(SFT_TRAINING_DATA,"r",encoding="utf-8") as f:
        json_data = json.load(f)

    return json_data


def disable_thinking(data, tokenizer):
    """显式应用 Qwen3 非思考模板，并保留 prompt-completion 训练结构。"""
    formatted_data = []

    for sample in data:
        prompt_messages = sample["prompt"]
        completion_messages = sample["completion"]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            prompt_messages + completion_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        if not full_text.startswith(prompt_text):
            raise ValueError("关闭 thinking 后，completion 无法与 prompt 正确对齐")

        formatted_sample = dict(sample)
        formatted_sample["prompt"] = prompt_text
        formatted_sample["completion"] = full_text[len(prompt_text):]
        formatted_data.append(formatted_sample)

    return formatted_data

def main():

    has_device = torch.cuda.is_available()
    if not has_device:
        raise RuntimeError("QLoRA 4bit 训练需要 CUDA GPU")
    else:
        device_name = torch.cuda.get_device_name()
        print(f"Using GPU,name:{device_name}")
        major,minor = torch.cuda.get_device_capability()

        is_support_bf16 = True if major >= 8 and torch.cuda.is_bf16_supported() else False
        print(f"是否支持bf16:{is_support_bf16}")

        compute_type = torch.float16
        print("强制采用fp16训练")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=SMALL_MODEL_PATH,
        trust_remote_code=False,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantify_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_type,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=SMALL_MODEL_PATH,
        quantization_config=quantify_config,
        trust_remote_code=False,
        torch_dtype=compute_type,
        device_map = {"":0}
    )

    kwargs ={
        "use_reentrant":False
    }

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs=kwargs
    )

    # k-bit 准备会把 RMSNorm 提升为 FP32；恢复为 FP16，避免 Q/K/V
    # 变成 FP32 后无法使用 memory-efficient SDPA。
    rmsnorm_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3RMSNorm":
            module.to(dtype=compute_type)
            rmsnorm_count += 1
    input_embeddings = model.get_input_embeddings()
    input_embeddings.to(dtype=compute_type)
    print(f"Qwen3RMSNorm dtype={compute_type}, count={rmsnorm_count}")
    print(f"input_embeddings dtype={input_embeddings.weight.dtype}")

    lora_config = LoraConfig(
        bias="none",
        lora_alpha=16,
        r = 8,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        lora_dropout=0.05
    )

    peft_model = get_peft_model(
        model=model,
        peft_config=lora_config
    )

    # FP16 GradScaler 不能解缩放 BF16 梯度。LoRA 可训练权重统一保留为
    # FP32，前向结果仍会转换回模型计算精度，且更有利于训练稳定性。
    trainable_dtype_count = 0
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            trainable_dtype_count += 1
    print(f"trainable parameters dtype=torch.float32, count={trainable_dtype_count}")

    print(f"active_adapters={peft_model.active_adapters}")
    #检查lora层的注入情况

    adapter_layer_count = 0
    disabled_adapter_layer_count = 0

    for layer in peft_model.modules():
        if isinstance(layer,BaseTunerLayer):
            adapter_layer_count += 1

            if layer.disable_adapters:
                disabled_adapter_layer_count += 1

    if adapter_layer_count == 0 or disabled_adapter_layer_count:
        raise RuntimeError("LoRA 层未注入或仍处于禁用状态")

    all_data = disable_thinking(load_data(), tokenizer)

    train_data = [data for data in all_data if data.get("metadata", {}).get("label") == "train"]
    test_data = [data for data in all_data if data.get("metadata", {}).get("label") == "test"]

    if not train_data or not test_data:
        raise ValueError("train 或 test 数据集为空")

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    sft_config = SFTConfig(
        # ==================== 输出 ====================
        output_dir=str(OUTPUT_DIR),

        # ==================== 训练规模 ====================
        num_train_epochs=3.0,
        learning_rate=1e-4,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        # ==================== 精度与显存 ====================
        bf16=False,
        fp16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=kwargs,

        # QLoRA 常用优化器
        optim="paged_adamw_8bit",

        # ==================== 优化策略 ====================
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ==================== 数据处理 ====================
        max_length=2304,

        # prompt 不计算 loss，只计算 completion
        completion_only_loss=True,

        # 暂时不把多条样本拼进同一序列
        packing=False,

        # ==================== 验证与保存 ====================
        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=5,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ==================== 其他 ====================
        report_to="none",
        seed=42,
        data_seed=42,
    )

    sft_trainer =SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    # SFTTrainer 初始化后可能重新包装模型；必须对 Trainer 实际持有的
    # 可训练参数再次统一 dtype，并且要在优化器创建之前完成。
    for param in sft_trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    trainer_trainable_dtypes = Counter(
        str(param.dtype)
        for param in sft_trainer.model.parameters()
        if param.requires_grad
    )
    print(f"trainer trainable parameter dtypes={dict(trainer_trainable_dtypes)}")
    if set(trainer_trainable_dtypes) != {"torch.float32"}:
        raise RuntimeError(
            f"Trainer 中仍存在非 FP32 可训练参数: {dict(trainer_trainable_dtypes)}"
        )

    check_sft_trainer(sft_trainer)

    sft_trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sft_trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"最佳 LoRA adapter 和 tokenizer 已保存到：{OUTPUT_DIR}")


def check_sft_trainer(
        sft_trainer: SFTTrainer
):
    first_data =next(iter(sft_trainer.train_dataset))

    if first_data is None:
        raise ValueError("数据集未导入sft_trainer里")

    peft_batch = sft_trainer.data_collator([first_data])
    peft_batch = {
        key:value.to(sft_trainer.accelerator.device)
        for key, value in peft_batch.items()
    }

    print(f"check_batch_input_shape={tuple(peft_batch['input_ids'].shape)}")

    sft_trainer.model.zero_grad()
    with torch.enable_grad(), sft_trainer.compute_loss_context_manager():
        peft_loss, _ = sft_trainer.compute_loss(
            sft_trainer.model,
            peft_batch,
            return_outputs=True
        )

        if not torch.isfinite(peft_loss) :
            raise ValueError("peft_loss趋近于无限大，运行终止")

        print(f"peft_loss={peft_loss}")

    sft_trainer.accelerator.backward(peft_loss)

    grad_tensor_count = 0
    grad_element_count = 0
    missing_grad_params = []

    total_grad_sq = None
    grad_dtype_count = Counter()
    non_fp32_grad_names = []

    for name, param in sft_trainer.model.named_parameters():

        # 1. 冻结参数不参与训练，没有 grad 是正常的
        if not param.requires_grad:
            continue

        # 2. 可训练参数没有梯度，才值得检查
        if param.grad is None:
            missing_grad_params.append(name)
            continue

        grad = param.grad.detach()
        grad_dtype_count[str(grad.dtype)] += 1
        if grad.dtype != torch.float32:
            non_fp32_grad_names.append(f"{name}: {grad.dtype}")

        # 3. 检查 NaN / Inf
        if not torch.isfinite(grad).all():
            raise ValueError(
                f"参数 {name} 的梯度中出现 NaN 或 Inf，训练终止"
            )

        # 4. 统计有梯度的参数 Tensor 数量
        grad_tensor_count += 1

        # 5. 统计真正的梯度元素数量
        grad_element_count += grad.numel()

        # 6. 计算所有梯度元素平方和
        grad_sq = grad.float().pow(2).sum()

        if total_grad_sq is None:
            total_grad_sq = grad_sq
        else:
            total_grad_sq += grad_sq

    # 7. 检查是否根本没有梯度
    if grad_tensor_count == 0:
        raise ValueError("所有可训练参数都不存在梯度，训练终止")

    # 8. 检查部分可训练参数梯度缺失
    if missing_grad_params:
        raise ValueError(
            f"存在可训练参数没有梯度，共 {len(missing_grad_params)} 个：\n"
            + "\n".join(missing_grad_params[:20])
        )

    # 9. 总梯度 L2 norm
    total_grad_norm = torch.sqrt(total_grad_sq)

    if total_grad_norm.item() == 0:
        raise ValueError("所有可训练参数梯度均为 0，训练终止")

    print(
        f"total_grad_norm={total_grad_norm.item():.6f}, "
        f"grad_tensor_count={grad_tensor_count}, "
        f"grad_element_count={grad_element_count}"
    )
    print(f"check gradient dtypes={dict(grad_dtype_count)}")
    if non_fp32_grad_names:
        raise RuntimeError(
            "检查阶段仍出现非 FP32 可训练梯度:\n"
            + "\n".join(non_fp32_grad_names[:20])
        )

    # 上面的 backward 只用于健全性检查，不能把缩放后的检查梯度带进正式训练。
    sft_trainer.model.zero_grad(set_to_none=True)

    return True

if __name__ == "__main__":
    main()

