from LightGraph import run_other_pipeline
import os


def main():

    run_other_pipeline(
        document_url=(
            r"D:\pycharmcode\ElectricityLLM\LLM-data\Document\电力电容器与电抗器.pdf"
        ),

        entity_cache_url=(
            r"D:\pycharmcode\ElectricityLLM\LLM-data\cache\capacitor.json"
        ),

        entity_dict_cache_url=(
            r"D:\pycharmcode\ElectricityLLM\LLM-data\cache\capacitor-dict.json"
        ),

        relationship_cache_url=(
            r"D:\pycharmcode\ElectricityLLM\LLM-data"
            r"\cache\capacitor-relationship-dict.json"
        ),

        elasticsearch_uri="https://localhost:9200",
        elasticsearch_username=os.environ["ELASTICSEARCH_USERNAME"],
        elasticsearch_password=os.environ["ELASTICSEARCH_PASSWORD"],
        es_index="graph_index",
        chunk_size=50,
        embedding_batch_size=8,
        embedding_model_url=(
            r"D:\pycharmcode\ElectricityLLM"
            r"\LLM-data\Models\bge-m3"
        ),
        embedding_cache_url=(
            r"D:\pycharmcode\ElectricityLLM"
            r"\LLM-data\Models\huggingface-cache"
        ),
        embedding_backend="auto",
        auto_download_embedding_model=True,
        embedding_local_files_only=False,
        docling_cache_url=(
            r"D:\pycharmcode\ElectricityLLM"
            r"\LLM-data\Models\docling-cache"
        ),
        docling_model_url=(
            r"D:\pycharmcode\LightGraph\.docling-cache\models\huggingface"
        ),

        # 文档分块
        max_tokens=1024,
        llm_model="glm-5.1",

        # 大模型 thinking
        entity_enable_thinking=False,
        relationship_enable_thinking=False,

        # 是否跳过入库
        skip_es_import=False,
        skip_graph_import=False,
    )

if __name__ == "__main__":
    main()