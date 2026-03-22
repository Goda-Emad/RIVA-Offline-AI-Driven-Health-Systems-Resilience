# Dockerfile - RIVA Health Platform v4.2.1
FROM python:3.9-slim

# تثبيت متطلبات النظام
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات أولاً
COPY requirements-web.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements-web.txt

# نسخ المشروع بالكامل (باستخدام مجلد data الصحيح)
COPY web-app/ ./web-app/
COPY ai-core/ ./ai-core/
COPY business-intelligence/ ./business-intelligence/
COPY data/ ./data/

# إنشاء مجلد logs
RUN mkdir -p /app/logs

# متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV RIVA_BASE_DIR=/app
ENV PYTHONPATH=/app

# مسار العمل
WORKDIR /app/web-app/src

# فتح المنفذ
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# تشغيل التطبيق
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
