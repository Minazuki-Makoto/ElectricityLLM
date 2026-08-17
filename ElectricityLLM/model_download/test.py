from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

MODEL_PATH = r"D:\pycharmcode\ElectricityLLM\ElectricityLLM\model\Qwen2.5-1.5B-Instruct"

PROMPT = "你是一位电气工程的助手，负责解答用户的问题"

def load_model(path = MODEL_PATH):

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map = "auto",
        local_files_only=True
    )

    print("模型加载成功")

    model.eval()

    return tokenizer,model

def chat(tokenizer,model,query:str):

    if (query == None):
        return "所问问题不能为空"

    message = [{
        "role": "system",
        "content":PROMPT
    }]

    query_message = {
        "role":"user",
        "content":query
    }

    message.append(query_message)

    message = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    message_input = tokenizer(
        message,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **message_input,
            max_new_tokens = 256,
            do_sample = False
        )

    output_ids = generated_ids[0][
                 message_input["input_ids"].shape[1]:
                 ]

    answer = tokenizer.decode(
        output_ids,
        skip_special_tokens=True
    )

    return answer

if __name__ == "__main__":
    tokenizer,model = load_model()
    query = "什么是变压器"
    print(chat(tokenizer,model,query))
