#!/usr/bin/env bash
set -euo pipefail

input="${1:?Usage: import_mysql.sh <dump.sql>}"
: "${MYSQL_HOST:?MYSQL_HOST is required}"
: "${MYSQL_PORT:=3306}"
: "${MYSQL_USERNAME:?MYSQL_USERNAME is required}"
: "${MYSQL_PASSWORD:?MYSQL_PASSWORD is required}"

defaults_file="$(mktemp)"
trap 'rm -f "$defaults_file"' EXIT
chmod 600 "$defaults_file"
printf '[client]\nuser=%s\npassword=%s\nhost=%s\nport=%s\n' \
  "$MYSQL_USERNAME" "$MYSQL_PASSWORD" "$MYSQL_HOST" "$MYSQL_PORT" > "$defaults_file"

mysql --defaults-extra-file="$defaults_file" --default-character-set=utf8mb4 < "$input"
echo "MySQL import completed from $input"
