import json
import os

from elasticsearch import Elasticsearch
from typing import Any
from ElasticSearch.embedding import embed
import numpy as np
from pydantic import BaseModel
from ElasticSearch.Elastic_base import link


class valid_format(BaseModel):
    equipment: str
    metric :str


def parse_response(
        response:dict[str,dict[str,Any]]
) -> list[dict[str,Any]]:

    parsed_results = []
    for hit in response.get("hits", {}).get("hits", []):
        score = hit.get("_score", 0) or 0
        source = hit.get("_source",{})

        if not isinstance(source,dict):
            raise ValueError("the source format is not right")

        if source == {}:
            raise ValueError("the source can not be empty")

        chunk_id = source.get("chunk_id")
        source_text = source.get("source")
        title = source.get("title")
        chapter_title = source.get("chapter_title")
        section_title = source.get("section_title")
        third_title = source.get("third_title")
        fourth_title = source.get("fourth_title")

        text = source.get("text")
        description = source.get("description")
        plot_data = source.get("plot_data", [])

        parsed_results.append(
            {
                "chunk_id": chunk_id,
                "source": source_text,
                "title": title,
                "chapter_title": chapter_title,
                "section_title": section_title,
                "third_title": third_title,
                "fourth_title": fourth_title,
                "text": text,
                "description": description,
                "plot_data": plot_data,
                "score":score
            }
        )
    return parsed_results


def normalize_score(
        results:list[dict[str,Any]],
        weights:float
) -> list[dict[str,Any]]:
    max_score = max((result.get("score", 0) for result in results), default=0)
    if max_score <= 0:
        return results

    for result in results:
        result["score"] = result.get("score", 0) / max_score * weights
    return results

#先做外围（除plot_data）的匹配
def title_keyword_match(
        es: Elasticsearch,
        index:str,
        query:str,
        topk:int,
        title_weight:float
) ->list[dict[str,Any]]:

    match_field = [
        "title^3",
        "chapter_title^2.5",
        "section_title^2",
        "third_title^1.5",
        "fourth_title",
    ]

    response = es.search(
        index = index,
        query = {
            "multi_match" : {
                "query":query,
                "fields":match_field,
            }
        },
        size = topk,
        source_excludes=["description_embedding"]
    )

    parsed_response = parse_response(response)

    return normalize_score(
        parsed_response,
        title_weight
    )

def description_text_keyword_match(
        es: Elasticsearch,
        index:str,
        query:str,
        topk:int,
        text_weight:float
) ->list[dict[str,Any]]:

    match_field = [
        "description^2",
        "text",
    ]

    response = es.search(
        index = index,
        query = {
            "multi_match" : {
                "query":query,
                "fields":match_field,
            }
        },
        size = topk,
        source_excludes=["description_embedding"]
    )

    parsed_response = parse_response(response)

    return normalize_score(
        parsed_response,
        text_weight
    )


def description_vector_match(
        es: Elasticsearch,
        index:str,
        query_vector:list[float],
        topk:int,
        vector_weight:float
) ->list[dict[str,Any]]:

    response = es.search(
        index = index,
        knn = {
            "field":"description_embedding",
            "query_vector":query_vector,
            "k":topk,
            "num_candidates":max(5*topk,20)
        },
        size = topk,
        source_excludes=["description_embedding"]
    )

    parsed_response = parse_response(response)

    return normalize_score(
        parsed_response,
        vector_weight
    )

def outside_match(
        es: Elasticsearch,
        index:str,
        query:str,
        query_vector:list[float],
        topk:int,
        title_weight:float,
        text_weight:float,
        vector_weight:float,
):

    title_match_res = title_keyword_match(
        es=es,
        index=index,
        query=query,
        topk=topk,
        title_weight=title_weight,
    )

    description_text_match_res = description_text_keyword_match(
        es=es,
        index=index,
        query=query,
        topk=topk,
        text_weight=text_weight
    )

    description_vector_match_res = description_vector_match(
        es=es,
        index=index,
        query_vector=query_vector,
        topk=topk,
        vector_weight=vector_weight
    )

    sum_dict: dict[str, dict[str, Any]] = {}
    for res in (
        title_match_res
        + description_text_match_res
        + description_vector_match_res
    ):
        chunk_id = res.get("chunk_id", "")
        if not isinstance(chunk_id, str):
            continue
        if not chunk_id.strip():
            continue

        if chunk_id not in sum_dict:
            sum_dict[chunk_id] = res.copy()

        else:
            sum_dict[chunk_id]["score"] += res.get("score", 0)

    sorted_sum = sorted(
        sum_dict.values(),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    return sorted_sum[:topk]


#开始匹配内部的plot_data
def inside_match(
    waited_for_match:dict[str,Any],
    query:str,
    query_vector:list[float],
):

    plot_data = waited_for_match.get("plot_data",[])
    if not isinstance(plot_data,list):
        raise ValueError("the plot_data format is not right")

    if len(plot_data) == 0:
        return {}

    max_score = float("-inf")
    best_match = None

    normalized_query = query.strip().lower()
    for data in plot_data:
        if not isinstance(data, dict):
            raise ValueError("the data in plot_data format is not right")
        metric_name = str(data.get("metric", "")).strip()
        if metric_name.lower() == normalized_query:
            return data

    for data in plot_data:
        if not isinstance(data,dict):
            raise ValueError("the data in plot_data format is not right")

        metric_name = str(data.get("metric", "")).strip()
        if not metric_name:
            continue

        metric_vector = data.get("metric_embedding")
        if not isinstance(metric_vector, list) or not metric_vector:
            # 兼容尚未补充metric_embedding的旧索引数据。
            metric_vector = embed(metric_name)

        denominator = np.linalg.norm(query_vector) * np.linalg.norm(metric_vector)
        if denominator == 0:
            continue
        similarity = np.dot(query_vector, metric_vector) / denominator
        if similarity > max_score:
            max_score = similarity
            best_match = data

    return best_match


def plot_full_search(
        elasticsearch_uri:str,
        elasticsearch_username:str,
        elasticsearch_password:str,
        index:str,
        equipment:str,
        metric:str,
        topk:int
):
    try:

        equipment_vector = embed(equipment)
        metric_vector = embed(metric)

        es = link(
            elasticsearch_uris=elasticsearch_uri,
            elasticsearch_username=elasticsearch_username,
            elasticsearch_password=elasticsearch_password
        )

        results = outside_match(
            es=es,
            index=index,
            query=equipment,
            query_vector=equipment_vector,
            topk=topk,
            text_weight=0.2,
            title_weight=0.3,
            vector_weight=0.5
        )

        retrieved_data = []
        for res in results:
            best_match = inside_match(
                waited_for_match=res,
                query = metric,
                query_vector=metric_vector,
            )

            if best_match == {}:
                continue
            points = best_match.get("points", [])
            if not isinstance(points, list):
                raise ValueError("the points format is not right")

            if not points:
                continue

            entity = best_match.get("entity","")
            dimension = best_match.get("dimension","")
            value_unit = best_match.get("value_unit","")

            for point in points:
                if not isinstance(point, dict):
                    continue
                label = point.get("label", "")
                x = point.get("x", "")
                value = point.get("value", 0)
                retrieved_data.append(
                    {
                        "equipment":entity,
                        "label":label,
                        "dimension":dimension,
                        "value":value,
                        "value_unit":value_unit,
                        "x":x
                    }
                )

        return retrieved_data

    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError("the query format is not right") from e


if __name__ == "__main__":
    message = {
        "equipment":"不同种类变压器",
        "metric":"负载损耗"
    }

    equipment = message.get("equipment","")
    metric = message.get("metric","")

    retrieved_data = plot_full_search(
        elasticsearch_uri="https://localhost:9200",
        elasticsearch_username=os.environ["ELASTICSEARCH_USERNAME"],
        elasticsearch_password=os.environ["ELASTICSEARCH_PASSWORD"],
        index = "electricity-plot-data",
        equipment=equipment,
        metric=metric,
        topk = 10
    )

    for res in retrieved_data:
        print(res)
        print("===================================")
