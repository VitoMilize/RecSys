# Используем минимальный Python-образ
FROM python:3.6-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN gdown https://drive.google.com/uc?id=1FVBYQNA-Sh9Ly-c88hUWQQUlsf2U4itc && \
    unzip pretrained_checkpoint.zip -d pretrained_checkpoint && \
    rm pretrained_checkpoint.zip

# Копируем весь проект
COPY model.py .
COPY main.py .
COPY dataset dataset

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]