# core/summarize_logs.py
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_LINES = 200
KEYWORDS = (
    "ERROR", "Error", "FAILED", "FAIL", "Traceback",
    "AssertionError", "TypeError", "ValueError",
    "ModuleNotFoundError", "ImportError",
    "mypy:", "ruff:", "pytest", "npm ERR", "openapi"
)

def die(msg: str) -> None:
    raise SystemExit(f"[Summarize] {msg}")

def main() -> None:
    log = Path("ai_test_last.log")
    if not log.exists():
        die("ai_test_last.log not found")

    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()

    picked: list[str] = []
    for i, line in enumerate(lines):
        if any(k in line for k in KEYWORDS):
            start = max(0, i - 3)
            end = min(len(lines), i + 5)
            picked.extend(lines[start:end])

    # 重複除去・整形
    uniq = []
    seen = set()
    for l in picked:
        if l not in seen:
            seen.add(l)
            uniq.append(l)

    summary = uniq[-MAX_LINES:]

    out = Path("ai_failure_summary.md")
    out.write_text(
        "# AI Failure Summary\n\n"
        "以下は、CI失敗時の要点抽出です。\n"
        "修正はこの内容のみを前提に行ってください。\n\n"
        "```text\n" + "\n".join(summary) + "\n```\n",
        encoding="utf-8",
    )

    print(f"[OK] summarized -> {out}")

if __name__ == "__main__":
    main()
