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
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.getcwd(), "config.ini"), encoding="utf-8")

        self.parser = argparse.ArgumentParser(description="Kaggle submission generator")
        self.parser.add_argument(
            "--model",
            type=str,
            default="LOG_REG",
            choices=["LOG_REG"],
            help="Имя модели",
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
            default=None,
            help="Куда сохранить submission.csv",
        )
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()

        model_path = self.config.get(
            "LOG_REG", "model_path", fallback=os.path.join("experiments", "tfidf_log_reg.pkl")
        )
        input_path = args.input or self.config.get(
            "DATA", "test_csv", fallback=os.path.join("data", "test.csv")
        )
        output_path = args.output or self.config.get(
            "LOG_REG", "submission_path", fallback=os.path.join("experiments", "submission.csv")
        )

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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sub.to_csv(output_path, index=False)
        self.log.info(f"Saved submission to {output_path}")
        return os.path.isfile(output_path)


if __name__ == "__main__":
    Predictor().predict()
