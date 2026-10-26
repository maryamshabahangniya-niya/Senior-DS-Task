FROM python:3.12-slim

WORKDIR /pipeline

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --index-url https://pypi.org/simple \
    --default-timeout=120 \
    --retries=3 \
    --prefer-binary \
    -r requirements.txt

COPY . .