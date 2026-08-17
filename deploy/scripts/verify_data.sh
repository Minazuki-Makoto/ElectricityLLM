#!/usr/bin/env bash
set -euo pipefail

compose_file="${COMPOSE_FILE:-deploy/docker-compose.yml}"

echo "MySQL tables"
docker compose -f "$compose_file" exec -T mysql sh -c \
  'mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" -e "SHOW TABLES;"'

echo "Elasticsearch index counts"
docker compose -f "$compose_file" exec -T elasticsearch sh -c \
  'curl -fsS -u "elastic:$ELASTIC_PASSWORD" "http://127.0.0.1:9200/_cat/indices?h=index,docs.count"'

echo "Neo4j node and relationship counts"
docker compose -f "$compose_file" exec -T neo4j sh -c \
  'cypher-shell -u neo4j -p "${NEO4J_AUTH#*/}" "MATCH (n) RETURN count(n) AS nodes; MATCH ()-[r]->() RETURN count(r) AS relationships;"' \
  || echo "Run the Neo4j count queries manually if NEO4J_AUTH parsing differs in the selected image."

echo "MinIO buckets"
docker compose -f "$compose_file" run --rm minio-init
