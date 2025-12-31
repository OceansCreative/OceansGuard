# core/openapi_contract.py
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    p = Path("contracts/openapi.json")
    if not p.exists():
        print("[openapi_contract] contracts/openapi.json not found (skip).")
        return 0

    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if text == "":
        print("[openapi_contract] contracts/openapi.json is empty (skip as placeholder).")
        return 0

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[openapi_contract] invalid json: {e.msg} (line {e.lineno}, col {e.colno})")
        return 1

    # Minimal sanity: must have openapi or swagger
    if not (isinstance(obj, dict) and ("openapi" in obj or "swagger" in obj)):
        print("[openapi_contract] json loaded but missing 'openapi'/'swagger' key.")
        return 1

    print("[openapi_contract] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
