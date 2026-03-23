FROM python:3.9-slim

WORKDIR /action

COPY src/review.py /action/review.py

RUN pip install --no-cache-dir requests

ENTRYPOINT ["python", "/action/review.py"]
