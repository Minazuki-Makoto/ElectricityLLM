from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_PATH = Path(__file__).resolve().parents[2]

PRIMARY_MODEL_PATH = (
        PROJECT_PATH
        / "LLM-data"
        / "Models"
        / "LLM"
        / "merged"
        / "Qwen3-4B-Instruct-electricity-sft-v1"
)

ADAPTER_PATH = (
        PROJECT_PATH
        / "LLM-data"
        / "Models"
        / "LLM"
        / "adapters"
        / "qwen3-4b-electricity-sft-v2"
)

MERGED_MODEL_PATH = (
        PROJECT_PATH
        / "LLM-data"
        / "Models"
        / "LLM"
        / "merged"
        / "Qwen3-4B-Instruct-electricity-sft-v2"
)

def check_paths() -> None:
    if not PRIMARY_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"基座模型不存在：{PRIMARY_MODEL_PATH}")

    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter 配置不存在：{ADAPTER_PATH}")

    if not (ADAPTER_PATH / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter 权重不存在：{ADAPTER_PATH}")


def main() -> None:
    check_paths()

    # 合并时不要使用4bit量化加载。
    # 在第一张CUDA显卡上以FP16加载并融合。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用，请检查NVIDIA驱动以及当前PyTorch是否为CUDA版本"
        )

    cuda_device = torch.device("cuda:0")
    gpu_properties = torch.cuda.get_device_properties(cuda_device)
    compute_dtype = torch.float16

    print(f"基座模型：{PRIMARY_MODEL_PATH}")
    print(f"Adapter：{ADAPTER_PATH}")
    print(f"合并模型输出：{MERGED_MODEL_PATH}")
    print(f"融合设备：{gpu_properties.name}")
    print(f"显存容量：{gpu_properties.total_memory / 1024 ** 3:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL_PATH,
        trust_remote_code=False,
        use_fast=True,
    )

    primary_model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_MODEL_PATH,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    peft_model = PeftModel.from_pretrained(
        primary_model,
        ADAPTER_PATH,
        is_trainable=False,
    )
    peft_model.eval()

    print("开始合并 LoRA Adapter……")

    merged_model = peft_model.merge_and_unload(
        safe_merge=True,
        progressbar=True,
    )

    # 这里只检查名称中包含 lora 的参数。
    lora_remaining = [
        name
        for name, _ in merged_model.named_parameters()
        if "lora_" in name.lower()
    ]

    if lora_remaining:
        raise RuntimeError(
            f"合并后仍检测到 LoRA 参数：{lora_remaining[:10]}"
        )

    MERGED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True,
        max_shard_size="4GB",
    )


    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("合并完成。")
    print(f"模型保存位置：{MERGED_MODEL_PATH}")


if __name__ == "__main__":
    main()
