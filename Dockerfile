# Используем минимальный Python-образ
FROM python:3.6-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY model.py .
COPY main.py .
COPY dataset dataset
COPY pretrained_checkpoint pretrained_checkpoint

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]