# Dockerfile
FROM python:3.9-slim

# تثبيت متطلبات النظام
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات
COPY requirements-web.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements-web.txt

# نسخ المشروع بالكامل
COPY web-app/ ./web-app/
COPY ai-core/ ./ai-core/
COPY data-storage/ ./data-storage/

# تعيين مسار العمل
WORKDIR /app/web-app/src

# فتح المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
