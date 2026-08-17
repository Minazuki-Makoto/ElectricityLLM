from .skill.short_memory import get_short_memory

from .skill.memory_vector import (
    abstract_search,
    hybridSearch_all_chat,
    judge_index,
    link,
)
from ElasticSearch.embedding import embed
from typing import Any
import logging


logger = logging.getLogger(__name__)


def load_memory(
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
        user_id:int,
        query:str,
        history_chat:list[dict[str,Any]]
):
    short_memory = get_short_memory(history_chat)
    try:
        es = link(
            elasticsearch_uris,
            elasticsearch_username,
            elasticsearch_password,
        )
        judge_index(es)
        query_vector = embed(query)
        abstract_memory_result = abstract_search(
            es=es,
            query_vector=query_vector,
            user_id=user_id,
            query=query,
            topk=2
        )

        abstract_memory_result_query = "。".join(
            f"【历史摘要{index}】:{str(result.get("query", "")).strip()}"
            for index,result in enumerate(abstract_memory_result, start=1)
            if str(result.get("query", "")).strip()
        )

        history_memory = hybridSearch_all_chat(
            es=es,
            query_vector=query_vector,
            user_id=user_id,
            query=query,
            threshold=0.4,
            abstract_results=abstract_memory_result
        )

        history_memory_query = "。".join(
            f"【历史对话{index}】用户：{str(result.get('query', '')).strip()}\n"
            f"助手：{str(result.get('answer', '')).strip()}"
            for index,result in enumerate(history_memory, start=1)
        )

    except Exception as error:
        logger.warning("Vector memory unavailable; using short memory only: %s", error)
        abstract_memory_result_query = ""
        history_memory_query = ""

    return {
        "short_memory":short_memory,
        "abstract_memory":abstract_memory_result_query,
        "history_memory":history_memory_query,
    }






