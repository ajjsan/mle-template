import configparser
import os
import unittest
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import MultiModel

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = MultiModel()

    def test_log_reg(self):
        self.assertEqual(self.multi_model.log_reg(), True)
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), "experiments", "tfidf_log_reg.pkl")))


if __name__ == "__main__":
    unittest.main()
