# Mapper backend for Railway (schema loaded from MongoDB; no target_schema.json in image)
FROM python:3.11-slim

WORKDIR /app

# System deps for building Python packages (e.g. faiss-cpu, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only; target_schema is loaded from MongoDB at runtime
COPY backend ./backend

ENV PORT=8080
EXPOSE 8080

# Bind to 0.0.0.0 so the platform can reach the app
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
