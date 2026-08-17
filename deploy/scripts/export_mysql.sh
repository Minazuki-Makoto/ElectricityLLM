#!/usr/bin/env bash
set -euo pipefail

: "${MYSQL_HOST:?MYSQL_HOST is required}"
: "${MYSQL_PORT:=3306}"
: "${MYSQL_DATABASE:=scms}"
: "${MYSQL_USERNAME:?MYSQL_USERNAME is required}"
: "${MYSQL_PASSWORD:?MYSQL_PASSWORD is required}"

output="${1:-./backup/mysql/${MYSQL_DATABASE}-$(date +%Y%m%d-%H%M%S).sql}"
mkdir -p "$(dirname "$output")"
defaults_file="$(mktemp)"
trap 'rm -f "$defaults_file"' EXIT
chmod 600 "$defaults_file"
printf '[client]\nuser=%s\npassword=%s\nhost=%s\nport=%s\n' \
  "$MYSQL_USERNAME" "$MYSQL_PASSWORD" "$MYSQL_HOST" "$MYSQL_PORT" > "$defaults_file"

mysqldump --defaults-extra-file="$defaults_file" \
  --single-transaction --routines --triggers --events \
  --set-gtid-purged=OFF --default-character-set=utf8mb4 \
  --databases "$MYSQL_DATABASE" > "$output"

echo "MySQL export written to $output"
