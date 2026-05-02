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

Гиперпараметры и split лежат в `params.yaml`.

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
