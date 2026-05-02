import configparser
import os
import pickle
import sys
import traceback

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from logger import Logger

SHOW_LOG = True


class MultiModel:
    """
    Оставил имя класса MultiModel ради совместимости с шаблоном,
    но теперь это тренер одной модели: TF-IDF + LogisticRegression.
    """

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        self.project_path = os.path.join(os.getcwd(), "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.log_reg_path = os.path.join(self.project_path, "tfidf_log_reg.pkl")

        try:
            train_split = self.config["SPLIT_DATA"]["train_split_csv"]
            val_split = self.config["SPLIT_DATA"]["val_split_csv"]
        except Exception:
            self.log.error(
                "Не найден SPLIT_DATA в config.ini. Сначала запусти src/preprocess.py."
            )
            raise

        self.train_df = pd.read_csv(train_split)
        self.val_df = pd.read_csv(val_split)
        self.log.info("Trainer is ready")

    def log_reg(self, predict: bool = True) -> bool:
        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=200_000)),
                ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
            ]
        )

        try:
            X_train = self.train_df["SentimentText"].astype(str).fillna("")
            y_train = self.train_df["Sentiment"].astype(int)
            X_val = self.val_df["SentimentText"].astype(str).fillna("")
            y_val = self.val_df["Sentiment"].astype(int)

            pipe.fit(X_train, y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if predict:
            y_pred = pipe.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"val_accuracy={acc:.4f}")

        return self.save_model(
            model=pipe,
            path=self.log_reg_path,
            name="LOG_REG",
            params={
                "path": os.path.relpath(self.log_reg_path, os.getcwd()),
                "vectorizer": "tfidf(1-2grams, max_features=200000)",
                "model": "LogisticRegression(max_iter=2000)",
            },
        )

    def save_model(self, model, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        with open("config.ini", "w", encoding="utf-8") as configfile:
            self.config.write(configfile)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self.log.info(f"{path} is saved")
        return os.path.isfile(path)


if __name__ == "__main__":
    MultiModel().log_reg(predict=True)
