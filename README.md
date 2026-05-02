# mle-template
Classic MLE template with CI/CD pipelines

## DVC

Проект настроен как DVC pipeline:

- `preprocess` -> создаёт `data/train_split.csv` и `data/val_split.csv`
- `train` -> обучает `TF-IDF + LogisticRegression`, сохраняет модель и метрику
- `predict` -> генерирует `experiments/submission.csv`

Основные команды:

```powershell
python -m dvc repro
python -m dvc status
python -m dvc metrics show
```

Пути, split и гиперпараметры модели лежат в `config.ini`.

## FastAPI

Сервис поднимает обученную модель `TF-IDF + LogisticRegression` и даёт endpoint для инференса.
Swagger-документация доступна по `http://127.0.0.1:8000/docs`.

Локальный запуск:

```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Проверка health:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Предсказание для одного текста:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"text":"I am so happy today!"}'
```

Предсказание для списка текстов (batch):

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict-batch -ContentType "application/json" -Body '{"texts":["I am very happy today","I feel terrible"]}'
```

## Docker Compose

Запуск API в контейнере:

```powershell
docker compose up --build
```

## Docker image

Собрать образ локально:

```powershell
docker build -t mle-template-api:latest .
```

Образ ставит зависимости из `requirements-api.txt` (минимальный набор для сервиса).

Важно: в образ по умолчанию копируется обученная модель `experiments/tfidf_log_reg.pkl`.
Сначала обучи модель (например, `python -m dvc repro` или `python .\\src\\train.py`), иначе `GET /health` покажет `model_loaded=false`.

Проверка:

```powershell
docker run --rm -p 8000:8000 mle-template-api:latest
```

И открой `http://127.0.0.1:8000/docs`.
