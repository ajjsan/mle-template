FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Кешируем слой зависимостей
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-api.txt

# Код и конфигурация
COPY . /app

EXPOSE 8000

# Перед сборкой образа убедись, что есть обученная модель:
# experiments/tfidf_log_reg.pkl (например: python -m dvc repro)
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
