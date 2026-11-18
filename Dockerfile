# One container for free hosting (Render free plan): FastAPI backend and Streamlit UI together.
# Free-tier storage is wiped on restart, so start.sh re-seeds the library from data/samples.
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-prod.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt -r requirements-ui.txt

COPY --chown=user . .

# Cloud configuration. GEMINI_API_KEY is NOT set here: add it as an environment variable in the Render dashboard.
ENV LLM_PROVIDER=gemini \
    EMBEDDING_PROVIDER=gemini \
    GEMINI_EMBEDDING_MODEL=gemini-embedding-2 \
    RETRIEVAL__SCORE_THRESHOLD=0.70 \
    BACKEND_URL=http://127.0.0.1:8000 \
    LOG_LEVEL=INFO

# The platform sets $PORT (Render: 10000); start.sh serves the UI on it.
CMD ["bash", "deploy/start.sh"]
