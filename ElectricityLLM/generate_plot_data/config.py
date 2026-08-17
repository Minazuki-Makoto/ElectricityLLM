from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT_PATH = PROJECT_ROOT / "LLM-data" / "Document" / "真实电抗器技术参数.docx"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "LLM-data" / "plot_json_data"
DEFAULT_MODEL = "glm-5.1"
DEFAULT_MAX_CHARS = 1800
DEFAULT_RETRIES = 3
