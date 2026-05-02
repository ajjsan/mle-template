import configparser
import html
import os
import sys
import traceback

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from logger import Logger

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHOW_LOG = True
PARAMS_PATH = os.path.join(os.getcwd(), "params.yaml")


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # Kaggle dataset contains HTML escapes like &lt;
    text = html.unescape(text)
    return " ".join(text.strip().split())


class DataMaker:
    """
    Подготовка датасета для Kaggle Twitter Sentiment Analysis:
    - data/train.csv: ItemID, Sentiment, SentimentText
    - data/test.csv: ItemID, SentimentText
    """

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.data_dir = os.path.join(os.getcwd(), "data")
        self.train_path = os.path.join(self.data_dir, "train.csv")
        self.test_path = os.path.join(self.data_dir, "test.csv")

        self.train_split_path = os.path.join(self.data_dir, "train_split.csv")
        self.val_split_path = os.path.join(self.data_dir, "val_split.csv")

        self.config_path = os.path.join(os.getcwd(), "config.ini")
        self.config = configparser.ConfigParser()
        self.log.info("DataMaker is ready")

    def read_params(self) -> dict:
        if not os.path.isfile(PARAMS_PATH):
            return {}
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_data(self) -> bool:
        ok = os.path.isfile(self.train_path) and os.path.isfile(self.test_path)
        if not ok:
            self.log.error(
                "Не найдены data/train.csv и/или data/test.csv. Проверь папку data/."
            )
        return ok

    def split_data(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> bool:
        params = self.read_params()
        split_params = params.get("split", {})
        test_size = split_params.get("test_size", test_size)
        random_state = split_params.get("random_state", random_state)

        if not self.get_data():
            return False
        try:
            # Kaggle-датасет часто лежит в latin1/Windows-1252, поэтому делаем фолбэк.
            try:
                df = pd.read_csv(self.train_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(self.train_path, encoding="latin1")
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        required_cols = {"ItemID", "Sentiment", "SentimentText"}
        if not required_cols.issubset(set(df.columns)):
            self.log.error(f"train.csv должен содержать колонки {sorted(required_cols)}")
            return False

        df = df.copy()
        df["SentimentText"] = df["SentimentText"].map(_clean_text)

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["Sentiment"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df.to_csv(self.train_split_path, index=False)
        val_df.to_csv(self.val_split_path, index=False)

        # Пишем относительные пути, чтобы проект был переносимым
        rel = lambda p: os.path.relpath(p, os.getcwd())
        self.config["DATA"] = {"train_csv": rel(self.train_path), "test_csv": rel(self.test_path)}
        self.config["SPLIT_DATA"] = {
            "train_split_csv": rel(self.train_split_path),
            "val_split_csv": rel(self.val_split_path),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            self.config.write(f)

        self.log.info("Train/val split is ready")
        return os.path.isfile(self.train_split_path) and os.path.isfile(self.val_split_path)


if __name__ == "__main__":
    DataMaker().split_data()
