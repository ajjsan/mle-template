import argparse
import configparser
import os
import pickle
import sys
import traceback

import pandas as pd

from logger import Logger

SHOW_LOG = True


class Predictor:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        self.parser = argparse.ArgumentParser(description="Kaggle submission generator")
        self.parser.add_argument(
            "--model",
            type=str,
            default="LOG_REG",
            choices=["LOG_REG"],
            help="Какая модель из config.ini",
        )
        self.parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="Путь к test.csv (если не задан, берём из config.ini)",
        )
        self.parser.add_argument(
            "--output",
            type=str,
            default=os.path.join("experiments", "submission.csv"),
            help="Куда сохранить submission.csv",
        )
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()

        try:
            model_path = self.config[args.model]["path"]
        except Exception:
            self.log.error("Не найден путь к модели в config.ini. Сначала запусти src/train.py.")
            return False

        input_path = args.input or self.config.get("DATA", "test_csv", fallback=os.path.join("data", "test.csv"))

        try:
            model = pickle.load(open(model_path, "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        try:
            try:
                df_test = pd.read_csv(input_path, encoding="utf-8")
            except UnicodeDecodeError:
                df_test = pd.read_csv(input_path, encoding="latin1")
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if "ItemID" not in df_test.columns or "SentimentText" not in df_test.columns:
            self.log.error("test.csv должен содержать колонки ItemID, SentimentText")
            return False

        preds = model.predict(df_test["SentimentText"].astype(str).fillna(""))
        sub = pd.DataFrame({"ItemID": df_test["ItemID"], "Sentiment": preds.astype(int)})

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        sub.to_csv(args.output, index=False)
        self.log.info(f"Saved submission to {args.output}")
        return os.path.isfile(args.output)


if __name__ == "__main__":
    Predictor().predict()
