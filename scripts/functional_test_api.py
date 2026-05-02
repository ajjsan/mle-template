#!/usr/bin/env python3
"""
Функциональные проверки HTTP API без внешних зависимостей (stdlib).
Использование:
  python scripts/functional_test_api.py --base-url http://127.0.0.1:8000 --timeout 120
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple


def http_json(method: str, url: str, body: Optional[Dict] = None, timeout: int = 30) -> Tuple[int, Dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        data = payload
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.getcode()
            return status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return exc.code, payload


def wait_for_model(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_detail = None
    while time.time() < deadline:
        status, body = http_json("GET", f"{base_url.rstrip('/')}/health")
        if status == 200 and body.get("model_loaded") is True:
            return
        last_detail = body
        time.sleep(2)
    raise RuntimeError(f"/health: model not ready after {timeout_s}s. Last response: {last_detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=120, help="Ожидание готовности модели, секунды")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    wait_for_model(base, args.timeout)

    status, health = http_json("GET", f"{base}/health")
    assert status == 200, health
    assert health.get("model_loaded") is True, health

    status, pred = http_json("POST", f"{base}/predict", {"text": "I am very happy today"})
    assert status == 200, pred
    assert pred.get("sentiment") in (0, 1), pred

    status, batch = http_json(
        "POST",
        f"{base}/predict-batch",
        {"texts": ["I love this", "This is terrible"]},
    )
    assert status == 200, batch
    preds = batch.get("predictions", [])
    assert len(preds) == 2, batch

    print("OK: functional API checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
