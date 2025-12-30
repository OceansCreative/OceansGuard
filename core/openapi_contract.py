# core/openapi_contract.py
from __future__ import annotations

import json
import sys
import time
import urllib.request
import subprocess
from pathlib import Path


OPENAPI_URL = "http://127.0.0.1:8000/openapi.json"
TIMEOUT_SEC = 15


def die(msg: str) -> None:
    raise SystemExit(f"[OpenAPI] {msg}")


def run(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def fetch_openapi(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def normalize(spec: dict) -> dict:
    """
    FastAPI の自動生成で揺れる部分を正規化
    """
    spec = dict(spec)
    spec.pop("servers", None)
    info = spec.get("info", {})
    info.pop("version", None)
    info.pop("title", None)
    return spec


def main() -> None:
    repo = Path(".").resolve()
    contract = repo / "contracts" / "openapi.json"

    if not contract.exists():
        die("contracts/openapi.json not found")

    # 起動（app.main:app を標準とする）
    proc = run([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"])
    try:
        # 起動待ち
        for _ in range(TIMEOUT_SEC):
            try:
                live = fetch_openapi(OPENAPI_URL)
                break
            except Exception:
                time.sleep(1)
        else:
            die("FastAPI did not start")

        expected = json.loads(contract.read_text(encoding="utf-8"))

        live_n = normalize(live)
        expected_n = normalize(expected)

        if live_n != expected_n:
            die("OpenAPI contract changed")

        print("[OK] OpenAPI contract unchanged")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
