FROM python:3.9-slim

# Устанавливаем git и очищаем кэш для уменьшения размера образа
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /action

COPY src/review.py /action/review.py

RUN pip install --no-cache-dir requests

ENTRYPOINT ["python", "/action/review.py"]
