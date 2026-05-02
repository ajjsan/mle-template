import configparser
import json
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
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.getcwd(), "config.ini"), encoding="utf-8")

        self.project_path = os.path.join(os.getcwd(), "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.log_reg_path = os.path.normpath(
            self.config.get(
                "LOG_REG",
                "model_path",
                fallback=os.path.join(self.project_path, "tfidf_log_reg.pkl"),
            )
        )
        self.metrics_path = os.path.normpath(
            self.config.get(
                "LOG_REG",
                "metrics_path",
                fallback=os.path.join(self.project_path, "metrics.json"),
            )
        )

        train_split = os.path.normpath(
            self.config.get(
                "SPLIT_DATA", "train_split_csv", fallback=os.path.join("data", "train_split.csv")
            )
        )
        val_split = os.path.normpath(
            self.config.get(
                "SPLIT_DATA", "val_split_csv", fallback=os.path.join("data", "val_split.csv")
            )
        )
        self.train_df = pd.read_csv(train_split)
        self.val_df = pd.read_csv(val_split)
        self.log.info("Trainer is ready")

    def log_reg(self, predict: bool = True) -> bool:
        pipe = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(
                            self.config.getint("TFIDF", "ngram_min", fallback=1),
                            self.config.getint("TFIDF", "ngram_max", fallback=2),
                        ),
                        max_features=self.config.getint("TFIDF", "max_features", fallback=200_000),
                        min_df=self.config.getint("TFIDF", "min_df", fallback=1),
                        max_df=self.config.getfloat("TFIDF", "max_df", fallback=1.0),
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.config.getint("LOG_REG", "max_iter", fallback=2000),
                        C=self.config.getfloat("LOG_REG", "c", fallback=1.0),
                        solver=self.config.get("LOG_REG", "solver", fallback="liblinear"),
                    ),
                ),
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
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                json.dump({"val_accuracy": float(acc)}, f, indent=2)

        return self.save_model(
            model=pipe,
            path=self.log_reg_path,
        )

    def save_model(self, model, path: str) -> bool:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self.log.info(f"{path} is saved")
        return os.path.isfile(path)


if __name__ == "__main__":
    MultiModel().log_reg(predict=True)
