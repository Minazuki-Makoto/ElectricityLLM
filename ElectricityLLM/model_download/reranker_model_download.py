from huggingface_hub import snapshot_download


MODEL_PATH = (
    r"D:\pycharmcode\ElectricityLLM"
    r"\LLM-data\Models\reranker-model"
)


def download_model() -> str:
    """从Hugging Face下载模型到指定目录。"""
    return snapshot_download(
        repo_id="BAAI/bge-reranker-v2-m3",
        local_dir=MODEL_PATH,
    )


if __name__ == "__main__":
    downloaded_path = download_model()
    print(f"模型下载完成：{downloaded_path}")