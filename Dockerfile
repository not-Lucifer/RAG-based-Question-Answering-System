# Hugging Face Space (Docker SDK): FastAPI backend and Streamlit UI in one container.
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

# Cloud configuration. GEMINI_API_KEY is NOT set here: add it as a Space secret.
ENV LLM_PROVIDER=gemini \
    EMBEDDING_PROVIDER=gemini \
    RETRIEVAL__SCORE_THRESHOLD=0.64 \
    BACKEND_URL=http://127.0.0.1:8000 \
    LOG_LEVEL=INFO

EXPOSE 7860
CMD ["bash", "deploy/huggingface/start.sh"]
