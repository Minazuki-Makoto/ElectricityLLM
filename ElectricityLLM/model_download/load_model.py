import os

# 1. 必须先设置环境变量，再导入 huggingface_hub

# 使用Clash代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# 避免其他代理配置发生冲突
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

# 强制使用Hugging Face官方地址，不再使用hf-mirror
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# Clash 代理下 Xet/CAS 大文件通道可能停在 0 字节，改用标准 HTTPS 下载。
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 将缓存放到D盘
os.environ["HF_HOME"] = (
    r"D:\pycharmcode\ElectricityLLM\LLM-data\cache\huggingface"
)
os.environ["HF_HUB_CACHE"] = (
    r"D:\pycharmcode\ElectricityLLM\LLM-data\cache\huggingface\hub"
)

# 下载大文件时适当延长超时时间
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

# 2. 环境变量设置完毕后再导入
from huggingface_hub import snapshot_download, constants


# 3. 模型信息
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

LOCAL_DIR = (
    r"D:\pycharmcode\ElectricityLLM\LLM-data"
    r"\Models\LLM\Qwen3-4B-Instruct-2507"
)

# 这是公开模型，不提供Token也能下载
TOKEN = os.getenv("HF_TOKEN")


def download_model():
    print(f"开始下载模型：{MODEL_ID}")
    print(f"实际下载端点：{constants.ENDPOINT}")
    print(f"保存位置：{LOCAL_DIR}")
    print(f"是否读取到Token：{TOKEN is not None}")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        token=TOKEN,
        max_workers=1
    )

    print("模型下载完成！")
    print(f"模型本地路径：{model_path}")


if __name__ == "__main__":
    download_model()
