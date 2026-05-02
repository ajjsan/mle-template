# DevOps
## Лабораторная работа №1 
- Выполнил: Хабибуллин Айсан

## Ссылки
- github: https://github.com/ajjsan/mle-template
- dockerhub: https://hub.docker.com/repository/docker/ajjsan/mle-template-api

## Детали реализации
### DVC

Проект настроен как DVC pipeline:

- `preprocess` -> создаёт `data/train_split.csv` и `data/val_split.csv`
- `train` -> обучает `TF-IDF + LogisticRegression`, сохраняет модель и метрику
- `predict` -> генерирует `experiments/submission.csv`

Пути, split и гиперпараметры модели лежат в `config.ini`.

### FastAPI

Сервис поднимает обученную модель  даёт endpoint для инференса.
Документация доступна по `http://127.0.0.1:8000/docs`.

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


### Docker image

Собрать образ локально:

```powershell
docker build -t mle-template-api:latest .
```

Образ ставит зависимости из `requirements-api.txt` (минимальный набор для сервиса).
Важно: в образ по умолчанию копируется обученная модель `experiments/tfidf_log_reg.pkl`.


### Jenkins CI/CD

В репозитории лежат пайплайны:

- `CI/Jenkinsfile` — сборка образа и push в Docker Hub (PR и/или обычные сборки веток — см. параметры `ONLY_*`)
- `CD/Jenkinsfile` — pull образа, запуск контейнера и функциональные проверки (`scripts/functional_test_api.py`)
