from elasticsearch import Elasticsearch
import os


def link(
        elasticsearch_uris,
        elasticsearch_username,
        elasticsearch_password,
):
    verify_certs = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"
    options = {
        "basic_auth": (elasticsearch_username, elasticsearch_password),
        "request_timeout": 30,
        "verify_certs": verify_certs,
        "ssl_show_warn": verify_certs,
    }
    ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS", "").strip()
    if ca_certs:
        options["ca_certs"] = ca_certs
    return Elasticsearch(
        elasticsearch_uris,
        **options,
    )
