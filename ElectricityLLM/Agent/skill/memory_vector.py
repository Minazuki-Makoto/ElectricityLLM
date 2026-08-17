import logging
import os
from typing import Any
from elasticsearch import Elasticsearch

from ElasticSearch.embedding.query_embedded import embed

logger = logging.getLogger(__name__)

MEMORY_INDEX = os.getenv("ELASTICSEARCH_MEMORY_INDEX", "memory-history")

def generate_es_index():
    return MEMORY_INDEX

def link(
        uris,
        username,
        password
):
    verify_certs = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"
    options = {
        "basic_auth": (username, password),
        "request_timeout": 30,
        "verify_certs": verify_certs,
        "ssl_show_warn": verify_certs,
    }
    ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS", "").strip()
    if ca_certs:
        options["ca_certs"] = ca_certs
    return Elasticsearch(
        uris,
        **options,
    )

def judge_index(
        es:Elasticsearch,
):
    existed = es.indices.exists(
        index=generate_es_index()
    )

    if existed:
        es.indices.put_mapping(
            index=generate_es_index(),
            properties={
                "memory_type":{"type":"keyword"},
                "chat_ids":{"type":"keyword"},
                "status":{"type":"keyword"}
            }
        )
        logger.info(
            "索引已存在,直接读取/写入即可"
        )

    else:

        es.indices.create(
            index=generate_es_index(),
            mappings={
                "properties": {
                    "id":{
                        "type":"keyword"
                    },
                    "user_id":{
                        "type":"integer"
                    },
                    "session_id":{
                        "type":"integer"
                    },
                    "type":{
                        "type":"text"
                    },
                    "memory_type":{
                        "type":"keyword"
                    },
                    "chat_ids":{
                        "type":"keyword"
                    },
                    "status":{
                        "type":"keyword"
                    },
                    "query":{
                        "type":"text"
                    },
                    "answer":{
                        "type":"text"
                    },
                    "query_vector":{
                        "type":"dense_vector",
                        "dims":1024,
                        "similarity":"cosine",
                        "index":True
                    },
                    "answer_vector":{
                        "type":"dense_vector",
                        "dims":1024,
                        "similarity":"cosine",
                        "index":True
                    }
                }
            }
        )

        logger.info(
            "索引已创建完成，可以直接写入"
        )

def write_in_es(
        elasticsearch_uri:str,
        elasticsearch_username:str,
        elasticsearch_password:str,
        id:str,
        user_id:int,
        session_id:int,
        query:str,
        answer:str,
        type:str,
        chat_ids:list[int] | None = None,
        status:str = "COMPLETED"
):
    es = link(
        uris=elasticsearch_uri,
        username=elasticsearch_username,
        password=elasticsearch_password
    )

    judge_index(
        es=es
    )

    query_vector = embed(query)
    document = {
        "id":id,
        "user_id":user_id,
        "session_id":session_id,
        "type":type,
        "memory_type":type,
        "status":status,
        "query":query,
        "answer":answer,
        "query_vector":query_vector
    }
    if answer.strip():
        document["answer_vector"] = embed(answer)
    if chat_ids is not None:
        document["chat_ids"] = [str(chat_id) for chat_id in chat_ids]

    es_document_id = f"abstract:{id}" if type == "abstract" else id

    es.index(
        index=generate_es_index(),
        id=es_document_id,
        document=document
    )

    logger.info(
        "最新数据已写入"
    )

def normalize_answer(
        results:list[dict[str,Any]],
        weight:float
):
    if not results:
        return []

    max_score = -float("inf")
    for result in results:
        if result.get("score",0) > max_score:
            max_score = result.get("score",0)

    if max_score <= 0:
        return results

    for result in results:
        result["score"] = weight * result.get("score",0)/max_score

    return results

def parse_response(
        response:dict[str,dict[str,Any]]
):
    results = []
    for hit in response["hits"]["hits"]:
        if not isinstance(hit,dict):
            raise ValueError("there may be some format errors in the response")
        score = hit.get("_score",0)
        source = hit.get("_source")
        if not isinstance(source,dict):
            raise ValueError("there may be some format errors in the source of the response")

        results.append({
            "score":score,
            "chunk_id":source.get("id"),
            "user_id":source.get("user_id"),
            "type":source.get("type"),
            "memory_type":source.get("memory_type"),
            "chat_ids":source.get("chat_ids",[]),
            "session_id":source.get("session_id"),
            "query":source.get("query"),
            "answer":source.get("answer")
        })

    return results

def keyword_search(
        query:str,
        user_id:int,
        chat_ids:list[str],
        es:Elasticsearch,
        weight:float=0.4,
        topk:int=3,
) -> list[dict[str,Any]]:
    match_field = [
        "query",
        "answer"
    ]

    response=es.search(
        index=generate_es_index(),
        query={
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": match_field
                    }
                },
                "filter":[
                    {
                        "term":{
                            "user_id":user_id
                        }
                    },
                    {
                        "term":{
                            "memory_type":"chat"
                        }
                    },
                    {
                        "terms":{
                            "id":chat_ids
                        }
                    }
                ]
            }
        },

        size=topk
    )

    results = parse_response(response)

    return normalize_answer(results,weight)

def vector_search(
        query_vector:list[float],
        user_id:int,
        chat_ids:list[str],
        es:Elasticsearch,
        weight:float=0.4,
        topk:int=3
):
    response=es.search(
        index=generate_es_index(),
        knn={
            "field":"query_vector",
            "num_candidates":max(topk*10,20),
            "query_vector":query_vector,
            "k":topk,
            "filter":[
                {
                    "term":{
                        "user_id":user_id
                    }
                },
                {
                    "term":{
                        "memory_type":"chat"
                    }
                },
                {
                    "terms":{
                        "id":chat_ids
                    }
                },
                {
                    "term":{
                        "status":"COMPLETED"
                    }
                }
            ]
        },
        size=topk
    )

    results = parse_response(response)
    return normalize_answer(results,weight)

def abstract_search(
        query:str,
        query_vector:list[float],
        user_id:int,
        es:Elasticsearch,
        topk:int=2,
        threshold:float=0.4
):
    keyword_response=es.search(
        index=generate_es_index(),
        query={
            "bool":{
                "must":{
                    "match":{
                        "query":query
                    }
                },
                "filter":[
                    {
                        "term":{
                            "user_id":user_id
                        }
                    },
                    {
                        "term":{
                            "memory_type":"abstract"
                        }
                    }
                ]
            }
        },
        size=topk
    )

    vector_response=es.search(
        index=generate_es_index(),
        knn={
            "field":"query_vector",
            "num_candidates":max(topk*10,20),
            "query_vector":query_vector,
            "k":topk,
            "filter":[
                {
                    "term":{
                        "user_id":user_id
                    }
                },
                {
                    "term":{
                        "memory_type":"abstract"
                    }
                }
            ]
        },
        size=topk
    )

    keyword_results = normalize_answer(
        parse_response(keyword_response),
        0.4
    )
    vector_results = normalize_answer(
        parse_response(vector_response),
        0.6
    )

    combined_results = {}
    for result in keyword_results + vector_results:
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue
        if chunk_id not in combined_results:
            combined_results[chunk_id] = result
        else:
            combined_results[chunk_id]["score"] += result.get("score",0)

    sorted_results = sorted(
        combined_results.values(),
        key=lambda x:x.get("score",0),
        reverse=True
    )

    return [
        result
        for result in sorted_results
        if result.get("score",0) > threshold
    ][:topk]

def hybridSearch_all_chat(
        es:Elasticsearch,
        abstract_results:list[dict[str,Any]],
        user_id:int,
        query:str,
        query_vector:list[float],
        threshold:float=0.4,
):

    judge_index(es)



    chat_ids = []
    for abstract_result in abstract_results:
        for chat_id in abstract_result.get("chat_ids",[]):
            chat_id = str(chat_id)
            if chat_id and chat_id not in chat_ids:
                chat_ids.append(chat_id)

    if not chat_ids:
        return []

    new_combined_results = {}

    for result in keyword_search(query,user_id,chat_ids,es,weight=0.4,topk=3)+vector_search(query_vector,user_id,chat_ids,es,weight=0.6,topk=3):
        if not isinstance(result,dict):
            raise ValueError("there may be some format errors in the response")

        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue

        if chunk_id not in new_combined_results:
            new_combined_results[chunk_id] = result

        else:
            new_combined_results[chunk_id]["score"] += result.get("score",0)

    sorted_results = sorted(
        new_combined_results.values(),
        key=lambda x:x.get("score",0),
        reverse=True
    )

    filtered_results =[]
    for result in sorted_results:
        if result.get("score",0) <= threshold:
            continue

        filtered_results.append(result)

    return filtered_results

