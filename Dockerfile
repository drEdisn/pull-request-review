FROM python:3.9-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --no-create-home --gecos "" reviewer

WORKDIR /action

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/review.py review.py

USER reviewer

ENTRYPOINT ["python", "/action/review.py"]
