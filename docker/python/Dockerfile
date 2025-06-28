# Image for fastapi

FROM python:3.9

RUN apt-get update && apt-get install -y \
    build-essential \
    cargo \
    gcc \
    libffi-dev \
    libssl-dev \
    musl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./project/requirements.txt /app/requirements.txt

WORKDIR /app

RUN python -m venv .venv && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    . .venv/bin/activate

ENV PATH="/app/.venv/bin:$PATH"

CMD tail -f /dev/null
