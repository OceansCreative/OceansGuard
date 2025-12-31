# core/summarize_logs.py
from __future__ import annotations

from pathlib import Path


def main() -> int:
    p = Path("ai_test_last.log")
    if not p.exists():
        print("[summarize_logs] ai_test_last.log not found.")
        return 0

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-80:] if len(lines) > 80 else lines

    print("[summarize_logs] tail:")
    for ln in tail:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
