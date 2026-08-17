# Elasticsearch deployment assets

The single-node service is defined in `../docker-compose.yml` with a 1 GiB JVM heap.

Use `../scripts/elasticsearch_migrate.py` to export and import all four online indexes. The tool preserves portable index settings, mappings, document IDs and document sources. Do not initialize production indexes with guessed mappings when a verified export exists.
