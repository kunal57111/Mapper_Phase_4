FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST to avoid the 2 GB CUDA bundle
# that sentence-transformers would otherwise pull in.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build tools to shrink the image
RUN apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY backend ./backend

ENV PORT=8080
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
