FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY web-app/ ./web-app/
COPY ai-core/ ./ai-core/
COPY business-intelligence/ ./business-intelligence/
COPY data/ ./data/

RUN mkdir -p /app/logs /app/data-storage/databases

WORKDIR /app/web-app

ENV PYTHONUNBUFFERED=1
ENV RIVA_BASE_DIR=/app
ENV PYTHONPATH=/app:/app/web-app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
