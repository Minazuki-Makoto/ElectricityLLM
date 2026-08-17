from json import JSONDecodeError
from pathlib import Path
import sys
import torch
from typing import Any,Dict
import json
import os


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from ElasticSearch import full_find
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_PATH = Path(__file__).resolve().parents[2]

PRIMARY_MODEL_PATH = (
        ROOT_PATH
        / "LLM-data"
        / "Models"
        / "LLM"
        / "base"
        / "Qwen3-4B-Instruct-2507"
)

MERGED_MODEL_PATH = (
        ROOT_PATH
        / "LLM-data"
        / "Models"
        / "LLM"
        / "merged"
        / "Qwen3-4B-Instruct-electricity-sft-v2"
)

VERIFYING_DATASET = (
    ROOT_PATH
    / "LLM-data"
    /"verifying_data"
    /"verification_query_text_50.jsonl"
)

RECORD_SAVE_PATH = (
    ROOT_PATH
    / "LLM-data"
    /"verifying_data"
    /"verification_merged_answer_2.jsonl"
)

SYSTEM_PROMPT = (
    "你是电力系统领域知识助手。"
    "请严格依据给定的检索资料回答问题，"
    "不得编造资料中不存在的事实。"
    "如果资料不足，请明确说明无法从现有资料中确定。"
)


RETRIEVAL_TOPK = 10
CONTEXT_TOPK = 5


def load_model_on_gpu(model_path: Path):
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用，请检查NVIDIA驱动以及当前PyTorch是否为CUDA版本"
        )

    gpu_properties = torch.cuda.get_device_properties(0)
    print(f"加载模型：{model_path}")
    print(f"推理设备：{gpu_properties.name}")
    print(f"显存容量：{gpu_properties.total_memory / 1024 ** 3:.2f} GB")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.float16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        local_files_only=True,
    )

    model.eval()
    return model, tokenizer


def load_base_model():
    return load_model_on_gpu(PRIMARY_MODEL_PATH)

def load_merged_model():
    return load_model_on_gpu(MERGED_MODEL_PATH)

def load_jsonl():
    queries = []
    with open(VERIFYING_DATASET, "r",encoding="utf-8") as f:
        for line_count , line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                json_dict = json.loads(line)

                query = json_dict.get("query")

                if not isinstance(query, str) or not query.strip():
                    raise ValueError(
                        f"第 {line_count} 行的 query 必须是非空字符串"
                    )

                queries.append(query.strip())

            except JSONDecodeError as e:
                raise ValueError(
                    f"第{line_count}行发生json格式错误{e}"
                )from e

    return queries

def package_message(
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        elastic_index,
        query,
        document_title_weight,
        title_weight,
        text_weight,
        vector_weight,
        description_text_weight,
        topk,
        score_threshold=None,
)->list[Dict[str, Any]]:

    message = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }
    ]

    retrieved = full_find(
        elasticsearch_uris=elasticsearch_uris,
        elasticsearch_username=elasticsearch_username,
        elasticsearch_password=elasticsearch_password,
        elastic_index=elastic_index,
        query=query,
        document_title_weight=document_title_weight,
        title_weight=title_weight,
        text_weight=text_weight,
        vector_weight=vector_weight,
        description_text_weight=description_text_weight,
        topk=topk,
        score_threshold=score_threshold,
    ) or []

    retrieved_text = ""
    for retrieved_info in retrieved[:CONTEXT_TOPK]:
        if not retrieved_info.get("text",""):
            continue

        retrieved_text += retrieved_info["text"]
        retrieved_text += "\n"

    user_query = f"""【检索到的资料】
{retrieved_text}

【用户问题】
{query}""".strip()

    message.append(
        {
            "role":"user",
            "content":user_query
        }
    )

    return message


def get_answer(
        model,
        tokenizer,
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        elastic_index,
        query,
        document_title_weight,
        title_weight,
        text_weight,
        vector_weight,
        description_text_weight,
        topk,
        score_threshold=None,
)->str:
    print("开始检索资料")
    message = package_message(
        elasticsearch_uris=elasticsearch_uris,
        elasticsearch_username=elasticsearch_username,
        elasticsearch_password=elasticsearch_password,
        elastic_index=elastic_index,
        query=query,
        document_title_weight=document_title_weight,
        title_weight=title_weight,
        text_weight=text_weight,
        vector_weight=vector_weight,
        description_text_weight=description_text_weight,
        topk=topk,
        score_threshold=score_threshold
    )

    print("资料已检索完毕，开始生成回答")

    prompt =tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    # 检索资料在前、用户问题在后；超长时优先保留后面的用户问题。
    tokenizer.truncation_side = "left"

    model_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    model_input = {
        key : value.to(model.device)
        for key,value in model_input.items()
    }
    input_length = model_input["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **model_input,
            use_cache = True,
            max_new_tokens = 512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated_ids = output_ids[0, input_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

def load_history():
    if not RECORD_SAVE_PATH.exists():
        return set()

    history_records = set()
    with open(RECORD_SAVE_PATH, "r",encoding="utf-8") as f:
        for line_count , line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                history_data = json.loads(line)
                history_query = history_data.get("query")
                if isinstance(history_query, str) and history_query.strip():
                    history_records.add(history_query.strip())

            except JSONDecodeError as e:
                raise ValueError(
                    f"json解析错误，错误原因:{e}"
                )from e

    return history_records

def main():
    if not VERIFYING_DATASET.is_file():
        raise FileNotFoundError(f"验证数据集不存在：{VERIFYING_DATASET}")

    RECORD_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    queries = load_jsonl()
    print("开始加载模型和分词器")
    model, tokenizer = load_merged_model()
    print("模型和分词器加载完毕")
    history_query = load_history()

    with open(RECORD_SAVE_PATH, "a", encoding="utf-8") as f:
        for query_index,query in enumerate(queries,start=1):

            if query in history_query:
                continue

            temporary = {}
            if not query.strip():
                continue
            answer = get_answer(
                model=model,
                tokenizer=tokenizer,
                elasticsearch_uris="https://localhost:9200",
                elasticsearch_username=os.environ["ELASTICSEARCH_USERNAME"],
                elasticsearch_password=os.environ["ELASTICSEARCH_PASSWORD"],
                elastic_index="electricity-infos",
                query=query,
                document_title_weight=0.10,
                title_weight=0.20,
                text_weight=0.02,
                vector_weight=0.60,
                description_text_weight=0.08,
                topk=30,
                score_threshold=0.30
            )

            if not answer.strip() :
                continue

            temporary["query"] = query
            temporary["answer"] = answer

            f.write(
                json.dumps(
                    temporary,
                    ensure_ascii=False,
                ) + "\n"
            )
            f.flush()

            print(f"第{query_index}个数据已写入")

if __name__ == "__main__":
    main()




