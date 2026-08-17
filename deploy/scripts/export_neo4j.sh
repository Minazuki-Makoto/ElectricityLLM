#!/usr/bin/env bash
set -euo pipefail

compose_file="${COMPOSE_FILE:-deploy/docker-compose.yml}"
backup_root="${BACKUP_ROOT:-/srv/electricity-llm/backup}"
mkdir -p "$backup_root/neo4j"

echo "Stopping Neo4j for a consistent Community Edition dump..."
docker compose -f "$compose_file" stop neo4j
docker compose -f "$compose_file" run --rm --no-deps \
  -v "$backup_root/neo4j:/backups" \
  neo4j neo4j-admin database dump neo4j --to-path=/backups --overwrite-destination=true
docker compose -f "$compose_file" start neo4j
echo "Neo4j dump written under $backup_root/neo4j"
