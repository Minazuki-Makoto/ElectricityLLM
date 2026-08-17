#!/usr/bin/env bash
set -euo pipefail

compose_file="${COMPOSE_FILE:-deploy/docker-compose.yml}"
backup_root="${BACKUP_ROOT:-/srv/electricity-llm/backup}"
dump_file="${1:?Usage: import_neo4j.sh <neo4j.dump>}"

mkdir -p "$backup_root/neo4j"
cp "$dump_file" "$backup_root/neo4j/neo4j.dump"
docker compose -f "$compose_file" stop neo4j
docker compose -f "$compose_file" run --rm --no-deps \
  -v "$backup_root/neo4j:/backups:ro" \
  neo4j neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true
docker compose -f "$compose_file" start neo4j
echo "Neo4j restore completed"
