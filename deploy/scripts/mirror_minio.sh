#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_MINIO_URL:?SOURCE_MINIO_URL is required}"
: "${SOURCE_MINIO_ACCESS_KEY:?SOURCE_MINIO_ACCESS_KEY is required}"
: "${SOURCE_MINIO_SECRET_KEY:?SOURCE_MINIO_SECRET_KEY is required}"
: "${TARGET_MINIO_URL:?TARGET_MINIO_URL is required}"
: "${TARGET_MINIO_ACCESS_KEY:?TARGET_MINIO_ACCESS_KEY is required}"
: "${TARGET_MINIO_SECRET_KEY:?TARGET_MINIO_SECRET_KEY is required}"

mc alias set source "$SOURCE_MINIO_URL" "$SOURCE_MINIO_ACCESS_KEY" "$SOURCE_MINIO_SECRET_KEY"
mc alias set target "$TARGET_MINIO_URL" "$TARGET_MINIO_ACCESS_KEY" "$TARGET_MINIO_SECRET_KEY"

for bucket in "${MINIO_PLOT_BUCKET:-plot}" "${MINIO_DOCUMENT_BUCKET:-contracts}"; do
  mc mb --ignore-existing "target/$bucket"
  mc mirror --overwrite --preserve "source/$bucket" "target/$bucket"
done

echo "MinIO mirror completed"
