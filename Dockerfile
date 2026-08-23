FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /app
COPY pyproject.toml ./
COPY structxai ./structxai
COPY cloud_service ./cloud_service
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /tmp/huggingface \
    && chown -R appuser:appuser /app /tmp/huggingface

USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "cloud_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
