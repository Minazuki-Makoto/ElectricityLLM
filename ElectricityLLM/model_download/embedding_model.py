import os
from pathlib import Path

# 必须在导入 huggingface_hub 前设置下载配置。
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault(
    "HF_HOME",
    r"D:\pycharmcode\ElectricityLLM\ElectricityLLM\model\hf_cache",
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    r"D:\pycharmcode\ElectricityLLM\ElectricityLLM\model\hf_cache\hub",
)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
# 代理环境中 Xet/CAS 大文件传输可能长时间停在 0 字节，改用普通 HTTP。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import constants, snapshot_download


MODEL_ID = "BAAI/bge-m3"
SAVE_ADDRESS = r"D:\pycharmcode\ElectricityLLM\ElectricityLLM\model\bge-m3"


def download_embedding_model(address: str = SAVE_ADDRESS) -> str:
    save_path = Path(address)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"开始下载编码模型：{MODEL_ID}")
    print(f"下载端点：{constants.ENDPOINT}")
    print(f"保存位置：{save_path}")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(save_path),
        token=os.getenv("HF_TOKEN"),
        max_workers=1,
        ignore_patterns=[
            "onnx/*",
            "imgs/*",
            "*.jpg",
            "*.webp",
        ],
    )

    print("模型下载完成。")
    print(f"模型本地路径：{model_path}")
    return model_path


if __name__ == "__main__":
    download_embedding_model()
