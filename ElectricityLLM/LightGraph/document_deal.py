from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT_PATH = Path(__file__).resolve().parents[3]
GRAPH_SYNC_PATH = Path(os.getenv("LIGHTGRAPH_PIPELINE_PATH", str(ROOT_PATH / "LightGraph")))
GRAPH_PYTHON_PATH = Path(os.getenv("LIGHTGRAPH_PYTHON_PATH", sys.executable))
TEMP_RESULT_PATH = Path(
    os.getenv("LIGHTGRAPH_TEMP_PATH", str(Path(tempfile.gettempdir()) / "LightGraph"))
)

def run_other_pipeline(
    document_url: str | Path | None = None,
    entity_cache_url: str | Path | None = None,
    entity_dict_cache_url: str | Path | None = None,
    relationship_cache_url: str | Path | None = None,
    elasticsearch_uri: str | None = None,
    elasticsearch_username: str | None = None,
    elasticsearch_password: str | None = None,
    es_index: str = "lightgraph_index",
    chunk_size: int = 50,
    embedding_batch_size: int = 8,
    embedding_model_url: str | Path | None = None,
    embedding_cache_url: str | Path | None = None,
    embedding_backend: str = "auto",
    auto_download_embedding_model: bool = True,
    embedding_local_files_only: bool = False,
    docling_cache_url: str | Path | None = None,
    docling_model_url: str | Path | None = None,
    llm_model: str | None = None,
    max_tokens: int = 512,
    entity_enable_thinking: bool = False,
    relationship_enable_thinking: bool = False,
    skip_es_import: bool = False,
    skip_graph_import: bool = False,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run LightGraph's ``run_pipeline`` in its own Python environment.

    Non-secret arguments are passed through a temporary JSON file. Database
    credentials are provided to the child process through environment
    variables and are never written to that file.
    """
    if not GRAPH_SYNC_PATH.is_dir():
        raise FileNotFoundError(f"LightGraph 项目目录不存在：{GRAPH_SYNC_PATH}")
    if not GRAPH_PYTHON_PATH.is_file():
        raise FileNotFoundError(
            f"LightGraph Python 解释器不存在：{GRAPH_PYTHON_PATH}"
        )
    if not (GRAPH_SYNC_PATH / "pipeline.py").is_file():
        raise FileNotFoundError(
            f"LightGraph 流水线模块不存在：{GRAPH_SYNC_PATH / 'pipeline.py'}"
        )

    if document_url is not None:
        document_path = Path(document_url).expanduser().resolve()
        if not document_path.is_file():
            raise FileNotFoundError(f"待处理文档不存在：{document_path}")
        document_url = document_path

    pipeline_arguments: dict[str, Any] = {
        "document_url": document_url,
        "entity_cache_url": entity_cache_url,
        "entity_dict_cache_url": entity_dict_cache_url,
        "relationship_cache_url": relationship_cache_url,
        "es_index": es_index,
        "chunk_size": chunk_size,
        "embedding_batch_size": embedding_batch_size,
        "embedding_model_url": embedding_model_url,
        "embedding_cache_url": embedding_cache_url,
        "embedding_backend": embedding_backend,
        "auto_download_embedding_model": auto_download_embedding_model,
        "embedding_local_files_only": embedding_local_files_only,
        "docling_cache_url": docling_cache_url,
        "docling_model_url": docling_model_url,
        "llm_model": llm_model,
        "max_tokens": max_tokens,
        "entity_enable_thinking": entity_enable_thinking,
        "relationship_enable_thinking": relationship_enable_thinking,
        "skip_es_import": skip_es_import,
        "skip_graph_import": skip_graph_import,
    }

    # None means "use the default declared by LightGraph.run_pipeline".
    pipeline_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in pipeline_arguments.items()
        if value is not None
    }

    child_environment = os.environ.copy()
    child_environment["PYTHONUTF8"] = "1"
    child_environment["PYTHONFAULTHANDLER"] = "1"
    if elasticsearch_uri is not None:
        child_environment["ELASTICSEARCH_URI"] = elasticsearch_uri
    if elasticsearch_username is not None:
        child_environment["ELASTICSEARCH_USERNAME"] = elasticsearch_username
    if elasticsearch_password is not None:
        child_environment["ELASTICSEARCH_PASSWORD"] = elasticsearch_password

    child_code = """
import json
import sys
from pathlib import Path

print("[LightGraph 子进程] 已启动，正在导入 pipeline", flush=True)
from pipeline import run_pipeline
print("[LightGraph 子进程] pipeline 导入完成", flush=True)

config_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])

with config_path.open("r", encoding="utf-8") as config_file:
    pipeline_arguments = json.load(config_file)

print("[LightGraph 子进程] 开始执行 run_pipeline", flush=True)
result = run_pipeline(**pipeline_arguments)

with result_path.open("w", encoding="utf-8") as result_file:
    json.dump(result, result_file, ensure_ascii=False, indent=2)
"""

    TEMP_RESULT_PATH.mkdir(parents=True, exist_ok=True)

    # Keep each invocation isolated and clean up its files automatically.

    with tempfile.TemporaryDirectory(
        prefix="pipeline_",
        dir=TEMP_RESULT_PATH,
    ) as temp_directory:
        temp_path = Path(temp_directory)
        config_path = temp_path / "pipeline_config.json"
        result_path = temp_path / "pipeline_result.json"

        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(
                pipeline_arguments,
                config_file,
                ensure_ascii=False,
                indent=2,
            )

        try:
            print(
                f"[LightGraph] 正在启动子进程：{GRAPH_PYTHON_PATH}",
                flush=True,
            )
            command = [
                str(GRAPH_PYTHON_PATH),
                "-c",
                child_code,
                str(config_path),
                str(result_path),
            ]
            subprocess.run(
                command,
                cwd=str(GRAPH_SYNC_PATH),
                env=child_environment,
                check=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"LightGraph 流水线运行超时（timeout={timeout} 秒）"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"LightGraph 流水线运行失败，退出码：{exc.returncode}"
                "；详细信息见当前控制台中该异常上方的子进程输出"
            ) from exc

        if not result_path.is_file():
            raise RuntimeError("LightGraph 已结束，但没有生成流水线结果文件")

        with result_path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)

    if not isinstance(result, dict):
        raise TypeError("LightGraph 流水线返回结果必须是字典")
    return result


__all__ = ["run_other_pipeline"]
