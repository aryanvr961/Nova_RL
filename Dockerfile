FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_home
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_home
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --disable-pip-version-check --retries 10 -r requirements.txt

COPY preload_models.py .
RUN python preload_models.py || true

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
