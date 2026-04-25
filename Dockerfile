# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copy and install Python dependencies first (layer caching) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app.py .

# ── Copy model artifacts ───────────────────────────────────────────────────────
COPY notebooks/models/ ./models/

# ── Set artifact path env variable ────────────────────────────────────────────
ENV ARTIFACTS_DIR=models/

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the API ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
