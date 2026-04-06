FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_home
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_home

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preload_models.py .
RUN python preload_models.py || true

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
