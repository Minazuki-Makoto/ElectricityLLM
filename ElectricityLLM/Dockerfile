FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ElectricityLLM/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY ElectricityLLM/ ./

RUN mkdir -p /app/logs /tmp/matplotlib \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:5000/health || exit 1

CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 4 --bind 0.0.0.0:${FLASK_PORT:-5000} --timeout ${GUNICORN_TIMEOUT_SECONDS:-300} --access-logfile - --error-logfile - app:app"]
