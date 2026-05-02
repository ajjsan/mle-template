import os
import pickle
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join("experiments", "tfidf_log_reg.pkl")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст твита")


class PredictResponse(BaseModel):
    sentiment: int
    label: str


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Список текстов для батч-предсказания")


class PredictBatchResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool


app = FastAPI(
    title="Twitter Sentiment API",
    description="API сервис для предсказания тональности текста с помощью TF-IDF + LogisticRegression.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена по пути '{MODEL_PATH}'. Сначала запусти обучение: python .\\src\\train.py"
        )

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def sentiment_to_label(sentiment: int) -> str:
    return "positive" if sentiment == 1 else "negative"


@app.get("/", tags=["service"])
def root():
    return {"message": "Twitter Sentiment API is running"}


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health_check():
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_path=MODEL_PATH,
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_sentiment(payload: PredictRequest):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле text не должно быть пустым")

    sentiment = int(model.predict([text])[0])
    return PredictResponse(
        sentiment=sentiment,
        label=sentiment_to_label(sentiment),
    )


@app.post("/predict-batch", response_model=PredictBatchResponse, tags=["inference"])
def predict_batch(payload: PredictBatchRequest):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cleaned_texts = [text.strip() for text in payload.texts]
    if any(not text for text in cleaned_texts):
        raise HTTPException(status_code=400, detail="В поле texts не должно быть пустых строк")

    sentiments = model.predict(cleaned_texts)
    predictions = [
        PredictResponse(sentiment=int(sentiment), label=sentiment_to_label(int(sentiment)))
        for sentiment in sentiments
    ]
    return PredictBatchResponse(predictions=predictions)
