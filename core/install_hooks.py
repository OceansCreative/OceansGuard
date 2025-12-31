# core/install_hooks.py
from __future__ import annotations

import argparse
from pathlib import Path


PRE_COMMIT = """#!/bin/sh
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" = "main" ]; then
  echo "ERROR: main への直接コミットは禁止です。ブランチを作ってください。"
  exit 1
fi
exit 0
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    hooks = repo / ".git" / "hooks"
    if not hooks.exists():
        raise SystemExit("ERROR: .git/hooks not found. Run inside a git repository.")

    target = hooks / "pre-commit"
    target.write_text(PRE_COMMIT, encoding="utf-8")

    # Try to set executable bit (harmless on Windows)
    try:
        target.chmod(0o755)
    except Exception:
        pass

    print(f"[OceansGuard] installed: {target}")


if __name__ == "__main__":
    main()
