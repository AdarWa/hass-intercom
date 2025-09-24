FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libportaudio2 \
        portaudio19-dev \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        git+https://github.com/cexen/py-simple-audio.git \
        sounddevice

CMD ["python", "-m", "server", "--host", "0.0.0.0", "--port", "8765"]
