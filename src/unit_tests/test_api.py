import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import api


class DummyModel:
    def predict(self, items):
        results = []
        for text in items:
            text_l = text.lower()
            if "happy" in text_l or "good" in text_l:
                results.append(1)
            else:
                results.append(0)
        return results


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        api.load_model.cache_clear()

    def tearDown(self):
        api.load_model.cache_clear()

    def test_health_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("status", body)
        self.assertIn("model_loaded", body)

    def test_predict_success(self):
        with patch("api.load_model", return_value=DummyModel()):
            response = self.client.post("/predict", json={"text": "I am very happy today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sentiment"], 1)
        self.assertEqual(body["label"], "positive")

    def test_predict_empty_text(self):
        with patch("api.load_model", return_value=DummyModel()):
            response = self.client.post("/predict", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("не должно быть пустым", response.json()["detail"])

    def test_predict_model_not_found(self):
        with patch("api.load_model", side_effect=FileNotFoundError("model not found")):
            response = self.client.post("/predict", json={"text": "hello"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("model not found", response.json()["detail"])

    def test_predict_batch_success(self):
        with patch("api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict-batch",
                json={"texts": ["I am happy", "I am sad"]},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["predictions"]), 2)
        self.assertEqual(body["predictions"][0]["label"], "positive")
        self.assertEqual(body["predictions"][1]["label"], "negative")

    def test_predict_batch_empty_item(self):
        with patch("api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict-batch",
                json={"texts": ["ok", "   "]},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("не должно быть пустых строк", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
