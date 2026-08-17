from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
from typing import Any
import os


def neo4j_link(
        uri,
        username,
        password
):
    driver = GraphDatabase.driver(
        uri=uri,
        auth=(username,password)
    )
    return driver

def generate_search_script(
        passing_nodes_name: list[str],
        passing_nodes_labels: list[str]
):
    if len(passing_nodes_name) != len(passing_nodes_labels):
        raise ValueError("途经节点名称和标签数量不一致")

    node_patterns = [
        "(a:$($start_node_label) {name: $start_node_name})"
    ]
    node_patterns.extend(
        (
            f"(passing_{index}:$($passing_node_label_{index}) "
            f"{{name: $passing_node_name_{index}}})"
        )
        for index in range(len(passing_nodes_name))
    )
    node_patterns.append(
        "(b:$($end_node_label) {name: $end_node_name})"
    )

    return (
        "MATCH path = "
        + "-[*1..10]->".join(node_patterns)
        + "\nRETURN nodes(path) AS nodes"
    )


def light_search(
        start_node_name:str,
        start_node_label:str,
        start_node_description:str,
        end_node_name:str,
        end_node_label:str,
        end_node_description:str,
        uri:str,
        username:str,
        password:str,
        passing_nodes:list[str],
        passing_nodes_label:list[str]
):

    driver = neo4j_link(uri=uri, username=username, password=password)
    search_script = generate_search_script(passing_nodes,passing_nodes_label)
    parameters = {
        "start_node_label": start_node_label,
        "start_node_name": start_node_name,
        "end_node_label": end_node_label,
        "end_node_name": end_node_name,
    }
    for index, (name, label) in enumerate(
        zip(passing_nodes, passing_nodes_label)
    ):
        parameters[f"passing_node_name_{index}"] = name
        parameters[f"passing_node_label_{index}"] = label

    try:
        records, summary, key = driver.execute_query(
            search_script,
            parameters_=parameters,
            database_=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

        passing_by_nodes_and_description = []
        passing_by_nodes_and_description.append(
            "首先，"
            f"{start_node_name}:{start_node_description}"
        )
        for record in records:
            for node in record["nodes"]:
                node_name = node.get("name", "")
                node_description = node.get("description", "")
                passing_by_nodes_and_description.append(
                    "其次，"
                    f"{node_name}:{node_description}"
                )

        passing_by_nodes_and_description.append(
            "最后，"
            f"{end_node_name}:{end_node_description}"
        )

        return passing_by_nodes_and_description

    finally:
        driver.close()

def es_link(
        uri,
        user_name,
        password
):

    verify_certs = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"
    options = {
        "basic_auth": (user_name, password),
        "request_timeout": 30,
        "verify_certs": verify_certs,
    }
    ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS", "").strip()
    if ca_certs:
        options["ca_certs"] = ca_certs
    return Elasticsearch(
        uri,
        **options,
    )

def parse_hit(
        response:dict[str,Any]
) -> list[dict[str,Any]]:

    parsed_results = []
    for hit in response["hits"]["hits"]:
        score = hit.get("_score",0)
        source = hit.get("_source",{})

        if not isinstance(source,dict):
            raise ValueError("the source parsed has a error format")

        if source == {}:
            raise ValueError("the source can not be empty")

        parsed_results.append(
            {
                "entity":source.get("entity"),
                "score":score,
                "entity_type":source.get("entity_type"),
                "description":source.get("description")
            }
        )

    return parsed_results


def es_keyword_search(
        linker:Elasticsearch,
        index:str,
        query:str,
        field:str,
        topk:int
):

    response = linker.search(
        index=index,
        query={
            "match":
                {
                    field:{
                        "query": query
                    }
                }
        },
        size = topk,
        source_excludes=["description_embedding"]
    )

    return parse_hit(response)[0]


def full_search(
        elasticsearch_uri:str,
        elasticsearch_username:str,
        elasticsearch_password:str,
        graph_uri,
        graph_username:str,
        graph_password:str,
        index:str,
        node:list[str],
        score_threshold:float,
):
    es_linker = es_link(
        elasticsearch_uri,
        elasticsearch_username,
        elasticsearch_password
    )
    if len(node) == 0:
        return ""

    start_node = node[0]
    start_best_match_start_node = es_keyword_search(
        es_linker,
        index=index,
        query=start_node,
        field="entity",
        topk=1
    )

    if start_best_match_start_node["score"]<score_threshold:
        return ""

    start_node_name = start_best_match_start_node.get("entity")
    start_node_label = start_best_match_start_node.get("entity_type")
    start_node_description = start_best_match_start_node.get("description")

    end_node = node[-1]
    end_best_match_end_node = es_keyword_search(
        es_linker,
        index=index,
        query=end_node,
        field="entity",
        topk=1
    )

    if end_best_match_end_node["score"]<score_threshold:
        return ""

    end_node_name = end_best_match_end_node.get("entity")
    end_node_label = end_best_match_end_node.get("entity_type")
    end_node_description = end_best_match_end_node.get("description")


    passing_node_name = []
    passing_node_label = []

    for i in range(1,len(node)-1):
        candidate_node = node[i]
        passing_best_match_node = es_keyword_search(
            es_linker,
            index=index,
            query=candidate_node,
            field="entity",
            topk=1
        )
        if passing_best_match_node["score"]<score_threshold:
            continue


        passing_node_name.append(passing_best_match_node.get("entity"))
        passing_node_label.append(passing_best_match_node.get("entity_type"))

    return ",".join(
        light_search(
            start_node_name,
            start_node_label,
            start_node_description,
            end_node_name,
            end_node_label,
            end_node_description,
            graph_uri,
            graph_username,
            graph_password,
            passing_node_name,
            passing_node_label
        )
    )

