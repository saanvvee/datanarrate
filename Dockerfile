# DataNarrate web app (FastAPI + static frontend over the src/ pipeline).
# Works on Render, Hugging Face Spaces and any Docker host; the host sets PORT (default 7860).
# LOKY_MAX_CPU_COUNT=1 keeps scikit-learn's n_jobs=-1 from starting one worker per host CPU on small instances.
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DATANARRATE_DB=/tmp/datanarrate.db \
    LOKY_MAX_CPU_COUNT=1 \
    OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY web ./web
COPY adult_income.csv mushroom.csv ./

# Hugging Face runs the container as a non-root user
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT}"]
