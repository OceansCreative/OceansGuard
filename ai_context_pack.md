# OceansGuard Context Pack

- generated_at: 2025-12-31T10:57:36
- repo: C:\Users\kazus\OneDrive\ドキュメント\GitHub\OceansGuard
- config_version: 1
- mode: normal

## Git

### git status --porcelain=v1
```text
 M .github/workflows/oceansguard.yml
 M README.md
 M core/aiguard.py
 M templates/.aiguard.yml
?? .aiguard.yml
?? ARCHITECTURE.md
?? SNAPSHOT.md
?? ai_check_report.json
?? ai_context_pack.md
?? contracts/README.md
?? core/install_hooks.py

```

### git diff
```diff
diff --git a/.github/workflows/oceansguard.yml b/.github/workflows/oceansguard.yml
index becac42..fcd38b0 100644
--- a/.github/workflows/oceansguard.yml
+++ b/.github/workflows/oceansguard.yml
@@ -1,4 +1,3 @@
-# .github/workflows/oceansguard.yml
 name: OceansGuard
 
 on:
@@ -32,21 +31,28 @@ jobs:
       - name: Upgrade pip
         run: python -m pip install --upgrade pip
 
-      # FastAPI / OpenAPI契約チェック用（未使用PJでも害なし）
-      - name: Install Python tooling (best-effort)
+      - name: Install tooling (best-effort)
         run: |
           pip install pyyaml || true
-          pip install uvicorn fastapi || true
           pip install ruff mypy pytest || true
+          pip install uvicorn fastapi || true
 
-      - name: OceansGuard init (idempotent)
+      - name: Resolve OceansGuard entry
+        id: og
         run: |
-          python core/aiguard.py init
+          if [ -f "tools/OceansGuard/core/aiguard.py" ]; then
+            echo "ENTRY=tools/OceansGuard/core/aiguard.py" >> $GITHUB_OUTPUT
+          elif [ -f "core/aiguard.py" ]; then
+            echo "ENTRY=core/aiguard.py" >> $GITHUB_OUTPUT
+          else
+            echo "OceansGuard entry not found" >&2
+            exit 1
+          fi
 
-      - name: OceansGuard pack
+      - name: OceansGuard init (idempotent)
         run: |
-          python core/aiguard.py pack
+          python "${{ steps.og.outputs.ENTRY }}" init --repo .
 
-      - name: OceansGuard check
+      - name: OceansGuard run (pack + check)
         run: |
-          python core/aiguard.py check
+          python "${{ steps.og.outputs.ENTRY }}" run --repo . --task "CI guard" --strict
diff --git a/README.md b/README.md
index 041756c..50e0786 100644
--- a/README.md
+++ b/README.md
@@ -1,54 +1,61 @@
+# README.md
 # OceansGuard
 
-OceansGuard は、生成AIによるコード変更を  
-**CI・契約・セキュリティで機械的に裁くためのガードレール**です。
-
-## 目的
-- AIにコードを書かせても事故らせない
-- 人が説明・確認・判断しなくてよい開発
-- どの言語・フレームワークでも共通運用
-
-## 基本思想
-- AIは「提案者」
-- 正しさは「テスト・契約・ポリシー」が決める
-- 通らない変更は採用されない
-
-## 使い方（各プロジェクト側）
-```bash
-python path/to/aiguard.py init
-python path/to/aiguard.py pack
-python path/to/aiguard.py check
-
-対応フェーズ
-
-開発前 / 開発途中 / 開発後 すべて対応
-
-
----
-
-## ③ あなたの「不可がほぼ無い」運用フロー（確定）
-**どの案件でもこれだけ**
-
-
-
-AIに投げる前 → ai:pack
-AI差分適用後 → ai:check
-通ったら → 採用
-
-
-- 考えない
-- 説明しない
-- レビューしない  
-
----
-
-## ④ 最初のGit操作（推奨）
-```bash
-git add .
-git commit -m "feat: initial OceansGuard core structure"
-git tag v0.1.0
-git push origin main --tags
-
-
-
-## Create by OceansCreative
\ No newline at end of file
+AI-assisted development guardrails for any repository.
+
+## What it solves
+- AI-generated changes that accidentally drop existing code
+- Lack of global context (only partial files shown)
+- Forgetfulness / inconsistent constraints across sessions
+- No test / lint guarantees
+- Secret leakage (keys/tokens) into commits
+- Risky full-rewrite changes
+
+## Core commands
+
+### init
+Create minimal guard files in target repo (idempotent; no overwrite).
+
+python core/aiguard.py init --repo .
+
+### pack
+Generate AI context pack (diff-first).
+```
+python core/aiguard.py pack --repo .
+```
+### check
+Run guard checks + configured project checks and write reports.
+```
+python core/aiguard.py check --repo .
+```
+### run
+Shortcut = pack + check.
+```
+python core/aiguard.py run --repo . --task "your task"
+```
+## Strict mode
+--strict makes guardrails non-negotiable:
+- requires PyYAML
+- fails if checks.commands is empty
+- fails if contracts/openapi.json is missing/empty
+```
+python core/aiguard.py run --repo . --task "CI guard" --strict
+```
+
+## Submodule usage (recommended)
+In your target repository:
+```
+git submodule add https://github.com/OceansCreative/OceansGuard.git tools/OceansGuard
+python tools/OceansGuard/core/aiguard.py init --repo .
+python tools/OceansGuard/core/aiguard.py run --repo . --task "初回ガード適用"
+```
+## Outputs
+- ai_context_pack.md: single file to paste into AI chat
+- ai_test_last.log: raw execution logs
+- ai_check_report.json: structured result for CI/PR gating
+
+## Git hooks (prevent committing to main)
+Install with:
+```
+python core/install_hooks.py --repo .
+```
\ No newline at end of file
diff --git a/core/aiguard.py b/core/aiguard.py
index dfad531..608d5df 100644
--- a/core/aiguard.py
+++ b/core/aiguard.py
@@ -2,156 +2,286 @@
 from __future__ import annotations
 
 import argparse
+import fnmatch
+import hashlib
 import json
 import os
-import shutil
+import re
 import subprocess
-import sys
 from dataclasses import dataclass
 from datetime import datetime
 from pathlib import Path
-from typing import Any
-
-
-PROJECT_NAME = "OceansGuard"
+from typing import Any, Dict, List, Optional, Tuple
 
 
+# =========================
+# Utilities
+# =========================
 def now_iso() -> str:
     return datetime.now().isoformat(timespec="seconds")
 
 
 def info(msg: str) -> None:
-    print(f"[{PROJECT_NAME}] {msg}")
+    print(f"[OceansGuard] {msg}")
 
 
 def warn(msg: str) -> None:
-    print(f"[{PROJECT_NAME}][WARN] {msg}")
+    print(f"[OceansGuard][WARN] {msg}")
 
 
 def die(msg: str, code: int = 1) -> None:
-    raise SystemExit(f"[{PROJECT_NAME}] {msg}")
+    raise SystemExit(f"[OceansGuard] {msg}")
 
 
-def copy_if_missing(src: Path, dst: Path) -> None:
-    if dst.exists():
-        info(f"[skip] exists: {dst}")
-        return
-    dst.parent.mkdir(parents=True, exist_ok=True)
-    shutil.copyfile(src, dst)
-    info(f"[create] {dst}")
+def sha256_text(s: str) -> str:
+    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
 
 
-def write_if_missing(dst: Path, content: str) -> None:
-    if dst.exists():
-        info(f"[skip] exists: {dst}")
-        return
-    dst.parent.mkdir(parents=True, exist_ok=True)
-    dst.write_text(content, encoding="utf-8")
-    info(f"[create] {dst}")
+def run_argv(argv, cwd=None):
+    import subprocess, os
 
+    p = subprocess.run(
+        argv,
+        cwd=cwd,
+        capture_output=True,
+        env=os.environ.copy(),
+    )
 
-def read_text_if_exists(p: Path) -> str | None:
-    if not p.exists():
-        return None
-    try:
-        return p.read_text(encoding="utf-8", errors="ignore")
-    except OSError:
-        return None
+    def _decode(b: bytes) -> str:
+        if not b:
+            return ""
+        try:
+            return b.decode("utf-8")
+        except UnicodeDecodeError:
+            return b.decode("utf-8", errors="replace")
 
+    p.stdout = _decode(p.stdout)
+    p.stderr = _decode(p.stderr)
+    return p
 
-def safe_json_load(p: Path) -> Any | None:
+
+
+def run_shell(cmd, cwd=None):
+    import subprocess, os
+
+    p = subprocess.run(
+        cmd,
+        cwd=cwd,
+        shell=True,
+        capture_output=True,
+        env=os.environ.copy(),
+    )
+
+    def _decode(b: bytes) -> str:
+        if not b:
+            return ""
+        try:
+            return b.decode("utf-8")
+        except UnicodeDecodeError:
+            return b.decode("utf-8", errors="replace")
+
+    p.stdout = _decode(p.stdout)
+    p.stderr = _decode(p.stderr)
+    return p
+
+
+def safe_read_text(p: Path, max_kb: int) -> str:
     try:
-        return json.loads(p.read_text(encoding="utf-8"))
-    except Exception:
-        return None
+        b = p.read_bytes()
+    except Exception as e:
+        return f"(failed to read: {e})\n"
+    if len(b) > max_kb * 1024:
+        return f"(skipped: too large {len(b)} bytes > {max_kb}KB)\n"
+    return b.decode("utf-8", errors="replace")
+
 
+def guard_root() -> Path:
+    # core/aiguard.py → OceansGuard/
+    return Path(__file__).resolve().parent.parent
 
-def try_import_yaml():
+
+def ensure_pyyaml(strict: bool) -> bool:
     try:
-        import yaml  # type: ignore
-        return yaml
+        import yaml  # noqa: F401
+        return True
     except Exception:
-        return None
+        if strict:
+            die("PyYAML is required in --strict mode. Install: pip install pyyaml")
+        warn("PyYAML not found. Some config-driven features may be skipped.")
+        return False
 
 
-@dataclass(frozen=True)
-class GuardOutput:
-    pack: str = "ai_context_pack.md"
-    audit: str = "CHANGELOG_AI.md"
-    testlog: str = "ai_test_last.log"
-
-
-@dataclass(frozen=True)
-class GuardConfig:
-    raw: dict[str, Any]
-    output: GuardOutput
-
-    @staticmethod
-    def load(repo: Path) -> "GuardConfig":
-        """
-        優先順位:
-        1) repo/.aiguard.yml
-        2) templates/.aiguard.yml を init がコピー済みならそれ
-        3) templates/.aiguard.yml を repo にコピーしてから読む
-        4) 最終的に空設定（最低限で通す）
-        """
-        cfg_path = repo / ".aiguard.yml"
-        if not cfg_path.exists():
-            # まず templates が同リポ内にある前提（OceansGuard自身）
-            # 他PJで submodule 利用のケースでも templates が来る想定
-            tpl = repo / "templates" / ".aiguard.yml"
-            if tpl.exists():
-                copy_if_missing(tpl, cfg_path)
-
-        if not cfg_path.exists():
-            warn(".aiguard.yml not found. Running with minimal defaults.")
-            raw = {}
-            return GuardConfig(raw=raw, output=GuardOutput())
-
-        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
-        yaml = try_import_yaml()
-        if yaml is None:
-            warn("PyYAML not installed. Some YAML features may not be parsed. "
-                 "Install: pip install pyyaml (CI already best-effort installs it).")
-            # 最低限: JSONとして読めるなら読む、無理なら空
-            raw = {}
-            return GuardConfig(raw=raw, output=GuardOutput())
+# =========================
+# Config (.aiguard.yml)
+# =========================
+@dataclass
+class ContextSmall:
+    include: List[str]
+
+
+@dataclass
+class ContextLarge:
+    roots: List[str]
+    exclude_dirs: List[str]
+    exclude_globs: List[str]
+    max_files: int
+    max_kb_each: int
+
+
+@dataclass
+class Evidence:
+    commands: List[str]
+
+
+@dataclass
+class Dlp:
+    enable: bool
+    block_on_detect: bool
+    mask: bool
+    allowlist_files: List[str]
+
+
+@dataclass
+class Guard:
+    forbid_full_rewrite: bool
+    allow_full_rewrite_globs: List[str]
+
+
+@dataclass
+class Checks:
+    commands: List[str]
+
+
+@dataclass
+class Output:
+    pack: str
+    audit: str
+    testlog: str
+    report_json: str
+
+
+@dataclass
+class Config:
+    version: int
+    context_small: ContextSmall
+    context_large: ContextLarge
+    evidence: Evidence
+    dlp: Dlp
+    guard: Guard
+    checks: Checks
+    output: Output
+
 
+def _dict_get(d: dict, key: str, default):
+    v = d.get(key, default)
+    return default if v is None else v
+
+
+def load_config(repo: Path, strict: bool) -> Config:
+    has_yaml = ensure_pyyaml(strict=strict)
+    cfg_path = repo / ".aiguard.yml"
+    if not cfg_path.exists():
+        die(".aiguard.yml not found. Run init first.")
+
+    raw: Dict[str, Any] = {}
+    if has_yaml:
         try:
-            raw = yaml.safe_load(text) or {}
+            import yaml  # type: ignore
+            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
         except Exception as e:
-            warn(f"Failed to parse .aiguard.yml: {e}")
+            if strict:
+                die(f"Failed to parse .aiguard.yml: {e}")
+            warn(f"Failed to parse .aiguard.yml; using minimal defaults. ({e})")
             raw = {}
+    else:
+        if strict:
+            die("Cannot read .aiguard.yml without PyYAML in strict mode.")
+
+    version = int(_dict_get(raw, "version", 1))
+
+    ctx = _dict_get(raw, "context", {})
+    small = _dict_get(ctx, "small", {})
+    large = _dict_get(ctx, "large", {})
+
+    context_small = ContextSmall(include=[str(x) for x in _dict_get(small, "include", [])])
+    context_large = ContextLarge(
+        roots=[str(x) for x in _dict_get(large, "roots", ["backend", "app", "src", "frontend"])],
+        exclude_dirs=[str(x) for x in _dict_get(large, "exclude_dirs", [
+            ".git", ".venv", "venv", "node_modules", "dist", "build", ".next",
+            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
+        ])],
+        exclude_globs=[str(x) for x in _dict_get(large, "exclude_globs", ["**/*.min.js", "**/*.map"])],
+        max_files=int(_dict_get(large, "max_files", 220)),
+        max_kb_each=int(_dict_get(large, "max_kb_each", 64)),
+    )
+
+    ev = _dict_get(raw, "evidence", {})
+    evidence = Evidence(commands=[str(x) for x in _dict_get(ev, "commands", [])])
+
+    dlp_raw = _dict_get(raw, "dlp", {})
+    dlp = Dlp(
+        enable=bool(_dict_get(dlp_raw, "enable", True)),
+        block_on_detect=bool(_dict_get(dlp_raw, "block_on_detect", True)),
+        mask=bool(_dict_get(dlp_raw, "mask", True)),
+        allowlist_files=[str(x) for x in _dict_get(dlp_raw, "allowlist_files", [])],
+    )
+
+    guard_raw = _dict_get(raw, "guard", {})
+    guard = Guard(
+        forbid_full_rewrite=bool(_dict_get(guard_raw, "forbid_full_rewrite", True)),
+        allow_full_rewrite_globs=[str(x) for x in _dict_get(guard_raw, "allow_full_rewrite_globs", [])],
+    )
+
+    checks_raw = _dict_get(raw, "checks", {})
+    checks = Checks(commands=[str(x) for x in _dict_get(checks_raw, "commands", [])])
+
+    out_raw = _dict_get(raw, "output", {})
+    output = Output(
+        pack=str(_dict_get(out_raw, "pack", "ai_context_pack.md")),
+        audit=str(_dict_get(out_raw, "audit", "CHANGELOG_AI.md")),
+        testlog=str(_dict_get(out_raw, "testlog", "ai_test_last.log")),
+        report_json=str(_dict_get(out_raw, "report_json", "ai_check_report.json")),
+    )
+
+    return Config(
+        version=version,
+        context_small=context_small,
+        context_large=context_large,
+        evidence=evidence,
+        dlp=dlp,
+        guard=guard,
+        checks=checks,
+        output=output,
+    )
+
+
+# =========================
+# init
+# =========================
+def copy_if_missing(src: Path, dst: Path) -> None:
+    if dst.exists():
+        info(f"skip exists: {dst}")
+        return
+    dst.parent.mkdir(parents=True, exist_ok=True)
+    dst.write_bytes(src.read_bytes())
+    info(f"create: {dst}")
+
+
+def write_if_missing(dst: Path, content: str) -> None:
+    if dst.exists():
+        info(f"skip exists: {dst}")
+        return
+    dst.parent.mkdir(parents=True, exist_ok=True)
+    dst.write_text(content, encoding="utf-8")
+    info(f"create: {dst}")
+
+
+def cmd_init(repo: Path) -> None:
+    tpl = guard_root() / "templates" / ".aiguard.yml"
+    if not tpl.exists():
+        die("templates/.aiguard.yml not found in OceansGuard")
+    copy_if_missing(tpl, repo / ".aiguard.yml")
 
-        out = raw.get("output") or {}
-        output = GuardOutput(
-            pack=str(out.get("pack", GuardOutput.pack)),
-            audit=str(out.get("audit", GuardOutput.audit)),
-            testlog=str(out.get("testlog", GuardOutput.testlog)),
-        )
-        return GuardConfig(raw=raw, output=output)
-
-
-def cmd_init(repo: Path) -> int:
-    """
-    init は「コピー専用」思想を維持。
-    - templates/.aiguard.yml を .aiguard.yml にコピー（上書きしない）
-    - ARCHITECTURE.md / SNAPSHOT.md を無ければ作成
-    - contracts/ を無ければ作成
-    - contracts/openapi.json が無ければ placeholder を作成（上書きしない）
-    """
-    here = Path(__file__).resolve()
-    guard_root = here.parent.parent  # OceansGuard/（このリポのルート想定）
-    templates = guard_root / "templates"
-
-    tpl_aiguard = templates / ".aiguard.yml"
-    if not tpl_aiguard.exists():
-        die("templates/.aiguard.yml not found")
-
-    # 1) .aiguard.yml
-    copy_if_missing(tpl_aiguard, repo / ".aiguard.yml")
-
-    # 2) ARCHITECTURE.md / SNAPSHOT.md
     write_if_missing(
         repo / "ARCHITECTURE.md",
         "# ARCHITECTURE\n\n"
@@ -169,229 +299,407 @@ def cmd_init(repo: Path) -> int:
         f"_generated by OceansGuard init @ {now_iso()}_\n",
     )
 
-    # 3) contracts/
     contracts = repo / "contracts"
-    if not contracts.exists():
-        contracts.mkdir(parents=True, exist_ok=True)
-        (contracts / "README.md").write_text(
-            "# Contracts\n\n"
-            "このディレクトリには、守るべき契約を置きます。\n\n"
-            "- OpenAPI（FastAPI の openapi.yaml / openapi.json）\n"
-            "- DB schema snapshot\n"
-            "- UI DTO / 型\n\n",
-            encoding="utf-8",
-        )
-        info(f"[create] {contracts}/README.md")
-    else:
-        info(f"[skip] exists: {contracts}/")
-
-    # 4) contracts/openapi.json placeholder（空ファイル対策：ただし上書きしない）
-    openapi = contracts / "openapi.json"
-    if not openapi.exists():
-        openapi.write_text(
-            json.dumps(
-                {
-                    "openapi": "3.0.3",
-                    "info": {"title": f"{PROJECT_NAME} Placeholder API", "version": "0.0.0"},
-                    "paths": {},
-                },
-                ensure_ascii=False,
-                indent=2,
-            ) + "\n",
-            encoding="utf-8",
-        )
-        info(f"[create] {openapi}")
+    contracts.mkdir(parents=True, exist_ok=True)
+    write_if_missing(
+        contracts / "README.md",
+        "# Contracts\n\n"
+        "このディレクトリには、守るべき契約（スキーマ/仕様）を置きます。\n\n"
+        "- OpenAPI: contracts/openapi.json（または openapi.yaml）\n"
+        "- DB schema snapshot\n"
+        "- DTO/型\n",
+    )
+    if not (contracts / "openapi.json").exists():
+        (contracts / "openapi.json").write_text("", encoding="utf-8")
+        info("create: contracts/openapi.json (empty)")
+
+    info("init completed")
+
+
+# =========================
+# pack (diff-first)
+# =========================
+def pack_git_section(repo: Path) -> str:
+    buf: List[str] = []
+    buf.append("## Git\n")
+
+    st = run_argv(["git", "status", "--porcelain=v1"], cwd=repo)
+    buf.append("\n### git status --porcelain=v1\n```text\n")
+    buf.append(st.stdout or "")
+    buf.append("\n```\n")
+
+    df = run_argv(["git", "diff"], cwd=repo)
+    diff_text = df.stdout or ""
+    buf.append("\n### git diff\n```diff\n")
+    if len(diff_text) > 200_000:
+        buf.append(diff_text[:200_000])
+        buf.append("\n... (truncated)\n")
     else:
-        info(f"[skip] exists: {openapi}")
+        buf.append(diff_text)
+    buf.append("\n```\n")
 
-    info("[OK] init completed")
-    return 0
+    return "".join(buf)
 
 
-def _ensure_guard_dir(repo: Path) -> Path:
-    d = repo / ".aiguard"
-    d.mkdir(parents=True, exist_ok=True)
-    return d
+def glob_files(root: Path, pattern: str) -> List[Path]:
+    return [p for p in root.glob(pattern) if p.is_file()]
 
 
-def _collect_context_pack(repo: Path, cfg: GuardConfig) -> str:
-    """
-    templates/.aiguard.yml の context.small.include を中心に “軽量” にまとめる。
-    無い場合は、代表ファイルだけを対象にする。
-    """
-    raw = cfg.raw
-    includes: list[str] = []
+def should_exclude(path: Path, repo: Path, exclude_dirs: List[str], exclude_globs: List[str]) -> bool:
+    rel = path.relative_to(repo).as_posix()
+    parts = set(Path(rel).parts)
+    if any(d in parts for d in exclude_dirs):
+        return True
+    for g in exclude_globs:
+        if fnmatch.fnmatch(rel, g):
+            return True
+    return False
 
-    ctx = raw.get("context") or {}
-    small = ctx.get("small") or {}
-    include = small.get("include")
-    if isinstance(include, list):
-        includes = [str(x) for x in include]
-    else:
-        includes = ["ARCHITECTURE.md", "SNAPSHOT.md", "README.md", "pyproject.toml", "package.json", "contracts/**"]
-
-    lines: list[str] = []
-    lines.append(f"# {PROJECT_NAME} Context Pack")
-    lines.append("")
-    lines.append(f"_generated @ {now_iso()}_")
-    lines.append("")
-
-    def add_file(path: Path) -> None:
-        rel = path.relative_to(repo)
-        text = read_text_if_exists(path)
-        if text is None:
-            return
-        # 過大防止：1ファイル最大 4000 文字
-        text = text[:4000]
-        lines.append(f"## {rel.as_posix()}")
-        lines.append("```")
-        lines.append(text.rstrip("\n"))
-        lines.append("```")
-        lines.append("")
-
-    for pat in includes:
-        if pat.endswith("/**") or pat.endswith("**"):
-            base = pat.replace("/**", "").replace("**", "").strip("/")
-            base_dir = repo / base
-            if base_dir.exists() and base_dir.is_dir():
-                for p in sorted(base_dir.rglob("*")):
-                    if p.is_file() and p.suffix.lower() in (".md", ".yml", ".yaml", ".json", ".toml", ".py", ".ts", ".tsx", ".js"):
-                        add_file(p)
-            continue
 
-        # glob対応
-        matched = list(repo.glob(pat))
-        if matched:
-            for p in matched:
-                if p.is_file():
-                    add_file(p)
+def collect_small(repo: Path, cfg: Config) -> List[Path]:
+    out: List[Path] = []
+    for pat in cfg.context_small.include:
+        out.extend(glob_files(repo, pat))
+    seen = set()
+    uniq: List[Path] = []
+    for p in sorted(out, key=lambda x: x.relative_to(repo).as_posix()):
+        rel = p.relative_to(repo).as_posix()
+        if rel in seen:
             continue
+        seen.add(rel)
+        uniq.append(p)
+    return uniq
 
-        # 通常ファイル
-        p = repo / pat
+
+def collect_changed_files(repo: Path) -> List[Path]:
+    cp = run_argv(["git", "diff", "--name-only"], cwd=repo)
+    files = []
+    for ln in (cp.stdout or "").splitlines():
+        ln = ln.strip()
+        if not ln:
+            continue
+        p = repo / ln
         if p.exists() and p.is_file():
-            add_file(p)
+            files.append(p)
+    return files
+
+
+def collect_large(repo: Path, cfg: Config) -> List[Path]:
+    files: List[Path] = []
+    roots = cfg.context_large.roots[:] if cfg.context_large.roots else ["."]
+    for r in roots:
+        base = (repo / r).resolve()
+        if not base.exists():
+            continue
+        for p in base.rglob("*"):
+            if not p.is_file():
+                continue
+            if should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs):
+                continue
+            files.append(p)
+    files = sorted(files, key=lambda x: x.relative_to(repo).as_posix())
+    if len(files) > cfg.context_large.max_files:
+        files = files[: cfg.context_large.max_files]
+    return files
+
+
+def evidence_section(repo: Path, cfg: Config) -> str:
+    buf: List[str] = []
+    buf.append("## Evidence\n")
+    for cmd in cfg.evidence.commands:
+        buf.append(f"\n### $ {cmd}\n")
+        cp = run_shell(cmd, cwd=repo)
+        buf.append("```text\n")
+        buf.append(cp.stdout or "")
+        buf.append("\n```\n")
+    return "".join(buf)
+
+
+def pack_files(repo: Path, files: List[Path], max_kb_each: int, title: str) -> str:
+    buf: List[str] = []
+    buf.append(f"## {title}\n")
+    buf.append("\n### File list\n```text\n")
+    for p in files:
+        buf.append(p.relative_to(repo).as_posix() + "\n")
+    buf.append("```\n\n")
+
+    for p in files:
+        rel = p.relative_to(repo).as_posix()
+        buf.append(f"### {rel}\n")
+        buf.append("```text\n")
+        buf.append(safe_read_text(p, max_kb=max_kb_each))
+        buf.append("\n```\n\n")
+    return "".join(buf)
+
+
+def cmd_pack(repo: Path, strict: bool) -> None:
+    cfg = load_config(repo, strict=strict)
+    out_path = repo / cfg.output.pack
+
+    small_files = collect_small(repo, cfg)
+
+    # diff-first: 変更ファイルを優先的に添付（除外規則も適用）
+    changed = [
+        p for p in collect_changed_files(repo)
+        if not should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs)
+    ]
+    # large は補助（上限付き）
+    large_files = collect_large(repo, cfg)
+
+    buf: List[str] = []
+    buf.append("# OceansGuard Context Pack\n\n")
+    buf.append(f"- generated_at: {now_iso()}\n")
+    buf.append(f"- repo: {repo}\n")
+    buf.append(f"- config_version: {cfg.version}\n")
+    buf.append(f"- mode: {'strict' if strict else 'normal'}\n\n")
+
+    buf.append(pack_git_section(repo))
+    buf.append(evidence_section(repo, cfg))
+
+    if changed:
+        buf.append(pack_files(repo, changed, max_kb_each=cfg.context_large.max_kb_each, title="Changed files (diff-first)"))
+    if small_files:
+        buf.append(pack_files(repo, small_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (small)"))
+    if large_files:
+        buf.append(pack_files(repo, large_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (large, capped)"))
+
+    out_path.write_text("".join(buf), encoding="utf-8")
+    info(f"pack written: {out_path}")
+
+
+# =========================
+# Guard: full rewrite detection
+# =========================
+def detect_full_rewrite(repo: Path, forbid: bool, allow_globs: List[str]) -> Optional[str]:
+    if not forbid:
+        return None
+
+    cp = run_argv(["git", "diff", "--numstat"], cwd=repo)
+    suspicious: List[str] = []
+
+    def _allowed(path_str: str) -> bool:
+        # numstat の file は基本 posix だが、念のため正規化
+        s = path_str.replace("\\", "/")
+        for g in allow_globs or []:
+            if fnmatch.fnmatch(s, g):
+                return True
+        return False
+
+    for ln in (cp.stdout or "").splitlines():
+        parts = ln.split("\t")
+        if len(parts) != 3:
+            continue
+        add_s, del_s, file_s = parts
+        if add_s == "-" or del_s == "-":
+            continue
+
+        if _allowed(file_s):
+            continue
+
+        try:
+            add_n = int(add_s)
+            del_n = int(del_s)
+        except ValueError:
+            continue
+
+        if (add_n + del_n) >= 800 and add_n >= 300 and del_n >= 300:
+            suspicious.append(f"{file_s} (add={add_n}, del={del_n})")
+
+    if suspicious:
+        return "Possible full rewrite detected:\n" + "\n".join(f"- {s}" for s in suspicious)
+    return None
+
+
+
+# =========================
+# DLP
+# =========================
+SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
+    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
+    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
+    ("GITHUB_TOKEN", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
+    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
+    ("JWT_LIKE", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
+    ("PASSWORD_ASSIGN", re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]")),
+]
+
+
+def is_allowlisted(rel_posix: str, allowlist: List[str]) -> bool:
+    for pat in allowlist:
+        if fnmatch.fnmatch(rel_posix, pat):
+            return True
+    return False
 
-    return "\n".join(lines).rstrip() + "\n"
 
+def scan_dlp(repo: Path, enable: bool, allowlist: List[str], mask: bool, strict: bool) -> List[str]:
+    if not enable:
+        return []
+    max_bytes = 256 * 1024  # 256KB
 
-def cmd_pack(repo: Path) -> int:
-    """
-    pack:
-    - .aiguard/ai_context_pack.md を生成（output.pack に従う）
-    - .aiguard/pack.json に簡易メタを書き出し
-    - ここでは “止めない” 方針（構成生成が目的）
-    """
-    cfg = GuardConfig.load(repo)
-    guard_dir = _ensure_guard_dir(repo)
+    cp = run_argv(["git", "ls-files"], cwd=repo)
+    hits: List[str] = []
 
-    pack_md = _collect_context_pack(repo, cfg)
-    (repo / cfg.output.pack).write_text(pack_md, encoding="utf-8")
-    info(f"[write] {cfg.output.pack}")
+    for rel in (cp.stdout or "").splitlines():
+        rel = rel.strip()
+        if not rel:
+            continue
+        if is_allowlisted(rel, allowlist):
+            continue
+        p = repo / rel
+        if not p.exists() or not p.is_file():
+            continue
+        try:
+            b = p.read_bytes()
+        except Exception:
+            continue
+        if len(b) > max_bytes:
+            continue
+        s = b.decode("utf-8", errors="ignore")
+        for name, pat in SECRET_PATTERNS:
+            m = pat.search(s)
+            if not m:
+                continue
+            sample = m.group(0)
+            if mask:
+                sample = f"<masked:{sha256_text(sample)[:12]}>"
+            hits.append(f"{name}: {rel}: {sample}")
+
+    # strict は「検知ルールのスキップ」ではなく、検知をより厳密にする余地（今は同等）
+    return hits
+
+
+# =========================
+# OpenAPI contract
+# =========================
+def openapi_contract_check(repo: Path, strict: bool) -> Optional[str]:
+    p = repo / "contracts" / "openapi.json"
+    if not p.exists():
+        if strict:
+            return "contracts/openapi.json is required in strict mode (can be empty only in normal mode)"
+        return None
 
-    meta = {
-        "project": PROJECT_NAME,
+    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
+    if not txt:
+        if strict:
+            return "contracts/openapi.json is empty in strict mode"
+        return None
+
+    try:
+        obj = json.loads(txt)
+    except Exception as e:
+        return f"contracts/openapi.json is not valid JSON: {e}"
+
+    if not isinstance(obj, dict):
+        return "contracts/openapi.json must be a JSON object"
+    if "openapi" not in obj and "swagger" not in obj:
+        return "contracts/openapi.json missing 'openapi' (or 'swagger') field"
+    return None
+
+
+# =========================
+# check (report json)
+# =========================
+def cmd_check(repo: Path, strict: bool) -> None:
+    cfg = load_config(repo, strict=strict)
+
+    report: Dict[str, Any] = {
+        "tool": "OceansGuard",
         "generated_at": now_iso(),
-        "output": dataclass_to_dict(cfg.output),
-        "note": "pack generates context artifacts. It should be non-blocking.",
+        "repo": str(repo),
+        "mode": "strict" if strict else "normal",
+        "checks": [],
+        "dlp_hits": [],
+        "guard": {"full_rewrite": None},
+        "openapi_contract": None,
+        "status": "pass",
     }
-    (guard_dir / "pack.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
-    info("[OK] pack completed")
-    return 0
-
 
-def _run_command(cmd: str, repo: Path) -> tuple[int, str]:
-    """
-    YAMLの checks.commands は shell 前提のものがあるため shell=True で実行。
-    """
-    p = subprocess.Popen(
-        cmd,
-        cwd=str(repo),
-        shell=True,
-        stdout=subprocess.PIPE,
-        stderr=subprocess.STDOUT,
-        text=True,
+    # 1) full rewrite guard
+    fr = detect_full_rewrite(repo, forbid=cfg.guard.forbid_full_rewrite, allow_globs=cfg.guard.allow_full_rewrite_globs)
+
+    if fr:
+        report["guard"]["full_rewrite"] = fr
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die(fr)
+
+    # 2) DLP
+    dlp_hits = scan_dlp(
+        repo,
+        enable=cfg.dlp.enable,
+        allowlist=cfg.dlp.allowlist_files,
+        mask=cfg.dlp.mask,
+        strict=strict,
     )
-    out, _ = p.communicate()
-    return p.returncode or 0, out or ""
-
-
-def _load_checks_commands(cfg: GuardConfig) -> list[str]:
-    raw = cfg.raw
-    checks = raw.get("checks") or {}
-    commands = checks.get("commands")
-    if isinstance(commands, list):
-        return [str(x) for x in commands]
-    return []
-
-
-def cmd_check(repo: Path) -> int:
-    """
-    check:
-    - templates/.aiguard.yml にある checks.commands を順に実行
-    - 失敗があれば exit 1（＝PRを止める）
-    - ただし summarize は || true が付いている想定なので失敗しても継続
-    """
-    cfg = GuardConfig.load(repo)
-    guard_dir = _ensure_guard_dir(repo)
-
-    cmds = _load_checks_commands(cfg)
-    if not cmds:
-        warn("No checks.commands found in .aiguard.yml. Nothing to run.")
-        # 何も無いなら成功扱い（ガードレールとしては最小）
-        return 0
-
-    log_lines: list[str] = []
-    log_lines.append(f"[{PROJECT_NAME}] check started @ {now_iso()}")
-    log_lines.append("")
-
-    failed = False
-
-    for i, cmd in enumerate(cmds, start=1):
-        log_lines.append(f"---")
-        log_lines.append(f"[{i}/{len(cmds)}] $ {cmd}")
-        rc, out = _run_command(cmd, repo)
-        log_lines.append(out.rstrip("\n"))
-        log_lines.append(f"[exit] {rc}")
-        log_lines.append("")
-
-        # shellで '|| true' を付けているものは rc=0 になる想定
-        if rc != 0:
-            failed = True
-
-    testlog_path = repo / cfg.output.testlog
-    testlog_path.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")
-    info(f"[write] {cfg.output.testlog}")
-
-    # 失敗ログは guard_dir にもコピーしておく（CIで参照しやすい）
-    (guard_dir / "ai_test_last.log").write_text(testlog_path.read_text(encoding="utf-8"), encoding="utf-8")
+    report["dlp_hits"] = dlp_hits
+    if dlp_hits and cfg.dlp.block_on_detect:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die("DLP detected potential secrets:\n" + "\n".join(f"- {h}" for h in dlp_hits))
+
+    # 3) OpenAPI contract
+    oc = openapi_contract_check(repo, strict=strict)
+    report["openapi_contract"] = oc
+    if oc:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die(oc)
+
+    # 4) checks.commands
+    log_path = repo / cfg.output.testlog
+    logs: List[str] = []
+    logs.append(f"[OceansGuard] check started @ {now_iso()}\n")
+
+    failed: List[Tuple[str, int]] = []
+
+    for i, cmd in enumerate(cfg.checks.commands, start=1):
+        cmd = (cmd or "").strip()
+        if not cmd:
+            continue
+        info(f"check[{i}] {cmd}")
+        cp = run_shell(cmd, cwd=repo)
+        entry = {
+            "index": i,
+            "command": cmd,
+            "exit_code": cp.returncode,
+        }
+        report["checks"].append(entry)
+
+        logs.append(f"\n=== check[{i}] {cmd} ===\n")
+        logs.append(cp.stdout or "")
+        logs.append(f"\n[exit_code] {cp.returncode}\n")
+
+        if cp.returncode != 0:
+            failed.append((cmd, cp.returncode))
+
+    log_path.write_text("".join(logs), encoding="utf-8")
+    info(f"testlog written: {log_path}")
+
+    # strict: checks.commands が空なら fail（「テスト保証」を強制）
+    if strict and not cfg.checks.commands:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die("checks.commands is empty in strict mode")
 
     if failed:
-        die("check failed (one or more commands returned non-zero).", code=1)
-    info("[OK] check completed")
-    return 0
-
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        summary = "\n".join([f"- ({rc}) {cmd}" for cmd, rc in failed])
+        die("check failed:\n" + summary)
 
-def cmd_run(repo: Path, task: str) -> int:
-    """
-    将来拡張用。現時点では最低限として、
-    - task が空なら check を実行
-    - task が init/pack/check のいずれかならそのまま実行
-    """
-    if not task:
-        return cmd_check(repo)
-    if task == "init":
-        return cmd_init(repo)
-    if task == "pack":
-        return cmd_pack(repo)
-    if task == "check":
-        return cmd_check(repo)
-    warn(f"Unknown task: {task}. Running check instead.")
-    return cmd_check(repo)
+    (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+    info(f"report written: {repo / cfg.output.report_json}")
+    info("check passed")
 
 
-def dataclass_to_dict(obj: Any) -> dict[str, Any]:
-    return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
+# =========================
+# run
+# =========================
+def cmd_run(repo: Path, task: str, strict: bool) -> None:
+    if task.strip():
+        info(f"run task: {task}")
+    cmd_pack(repo, strict=strict)
+    cmd_check(repo, strict=strict)
 
 
 def main() -> None:
@@ -399,20 +707,25 @@ def main() -> None:
     ap.add_argument("command", choices=["init", "pack", "check", "run"])
     ap.add_argument("--repo", default=".")
     ap.add_argument("--task", default="")
+    ap.add_argument("--strict", action="store_true", help="Do not allow skipping; fail if prerequisites missing.")
     args = ap.parse_args()
 
     repo = Path(args.repo).resolve()
 
     if args.command == "init":
-        raise SystemExit(cmd_init(repo))
+        cmd_init(repo)
+        return
     if args.command == "pack":
-        raise SystemExit(cmd_pack(repo))
+        cmd_pack(repo, strict=args.strict)
+        return
     if args.command == "check":
-        raise SystemExit(cmd_check(repo))
+        cmd_check(repo, strict=args.strict)
+        return
     if args.command == "run":
-        raise SystemExit(cmd_run(repo, args.task))
+        cmd_run(repo, task=args.task, strict=args.strict)
+        return
 
-    raise SystemExit(0)
+    die("unknown command")
 
 
 if __name__ == "__main__":
diff --git a/templates/.aiguard.yml b/templates/.aiguard.yml
index 6b1bdcc..a8dbaaf 100644
--- a/templates/.aiguard.yml
+++ b/templates/.aiguard.yml
@@ -21,19 +21,21 @@ context:
       - build
       - .next
       - __pycache__
+      - __pypackages__
       - .pytest_cache
       - .mypy_cache
       - .ruff_cache
     exclude_globs: ["**/*.min.js", "**/*.map"]
     max_files: 220
     max_kb_each: 64
-  evidence:
-    commands:
-      - git status --porcelain=v1 || true
-      - git diff || true
-      - python --version || true
-      - node --version || true
-      - npm --version || true
+
+evidence:
+  commands:
+    - git status --porcelain=v1 || true
+    - git diff || true
+    - python --version || true
+    - node --version || true
+    - npm --version || true
 
 dlp:
   enable: true
@@ -44,37 +46,33 @@ dlp:
     - "frontend/.env.example"
 
 guard:
-  max_files: 10
-  max_lines: 400
   forbid_full_rewrite: true
+  allow_full_rewrite_globs:
+    - "core/aiguard.py"
+    - ".github/workflows/oceansguard.yml"
 
 checks:
   commands:
     # =========================
-    # Backend (FastAPI / Python)
+    # Backend (Python)
     # =========================
-
-    # compileall: backend/app/src が存在する場合のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       sys.exit(subprocess.call([sys.executable,'-m','compileall',*targets]) if targets else 0)"
 
-    # ruff: インストール済み かつ 対象dir存在時のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       has=importlib.util.find_spec('ruff') is not None;
       sys.exit(subprocess.call([sys.executable,'-m','ruff','check',*targets]) if (has and targets) else 0)"
 
-    # mypy: インストール済み かつ 対象dir存在時のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       has=importlib.util.find_spec('mypy') is not None;
       sys.exit(subprocess.call([sys.executable,'-m','mypy',*targets]) if (has and targets) else 0)"
 
-    # pytest: tests が存在し、pytest導入済みの場合のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       has=importlib.util.find_spec('pytest') is not None;
@@ -82,34 +80,20 @@ checks:
       sys.exit(subprocess.call([sys.executable,'-m','pytest','-q']) if (has and has_tests) else 0)"
 
     # =========================
-    # Frontend (React / Node)
+    # Frontend (React)
     # =========================
-
-    # npm ci (or npm install): frontend が存在する場合のみ実行
     - >
       python -c "import os,sys,subprocess;
       d='frontend';
       sys.exit(0 if not os.path.isdir(d) else (subprocess.call('npm ci --silent', cwd=d, shell=True) if os.path.exists(os.path.join(d,'package-lock.json')) else subprocess.call('npm install --silent', cwd=d, shell=True)))"
 
-    # npm run build: frontend が存在する場合のみ実行
     - >
       python -c "import os,sys,subprocess;
       d='frontend';
       sys.exit(0 if not os.path.isdir(d) else subprocess.call('npm run build --silent', cwd=d, shell=True))"
 
-    # （既存の Python / React checks の下に追加）
-    - python core/openapi_contract.py
-
-    - >
-      python -c "import os,sys;
-      sys.exit(0 if not os.path.exists('contracts/openapi.json') else
-      __import__('subprocess').call([sys.executable,'core/openapi_contract.py']))"
-
-    # 失敗時ログ要約（check が失敗しても要約は残す）
-    - python core/summarize_logs.py || true
-
-
 output:
   pack: ai_context_pack.md
   audit: CHANGELOG_AI.md
   testlog: ai_test_last.log
+  report_json: ai_check_report.json

```
## Evidence

### $ git status --porcelain=v1 || true
```text
 M .github/workflows/oceansguard.yml
 M README.md
 M core/aiguard.py
 M templates/.aiguard.yml
?? .aiguard.yml
?? ARCHITECTURE.md
?? SNAPSHOT.md
?? ai_check_report.json
?? ai_context_pack.md
?? contracts/README.md
?? core/install_hooks.py

```

### $ git diff || true
```text
diff --git a/.github/workflows/oceansguard.yml b/.github/workflows/oceansguard.yml
index becac42..fcd38b0 100644
--- a/.github/workflows/oceansguard.yml
+++ b/.github/workflows/oceansguard.yml
@@ -1,4 +1,3 @@
-# .github/workflows/oceansguard.yml
 name: OceansGuard
 
 on:
@@ -32,21 +31,28 @@ jobs:
       - name: Upgrade pip
         run: python -m pip install --upgrade pip
 
-      # FastAPI / OpenAPI契約チェック用（未使用PJでも害なし）
-      - name: Install Python tooling (best-effort)
+      - name: Install tooling (best-effort)
         run: |
           pip install pyyaml || true
-          pip install uvicorn fastapi || true
           pip install ruff mypy pytest || true
+          pip install uvicorn fastapi || true
 
-      - name: OceansGuard init (idempotent)
+      - name: Resolve OceansGuard entry
+        id: og
         run: |
-          python core/aiguard.py init
+          if [ -f "tools/OceansGuard/core/aiguard.py" ]; then
+            echo "ENTRY=tools/OceansGuard/core/aiguard.py" >> $GITHUB_OUTPUT
+          elif [ -f "core/aiguard.py" ]; then
+            echo "ENTRY=core/aiguard.py" >> $GITHUB_OUTPUT
+          else
+            echo "OceansGuard entry not found" >&2
+            exit 1
+          fi
 
-      - name: OceansGuard pack
+      - name: OceansGuard init (idempotent)
         run: |
-          python core/aiguard.py pack
+          python "${{ steps.og.outputs.ENTRY }}" init --repo .
 
-      - name: OceansGuard check
+      - name: OceansGuard run (pack + check)
         run: |
-          python core/aiguard.py check
+          python "${{ steps.og.outputs.ENTRY }}" run --repo . --task "CI guard" --strict
diff --git a/README.md b/README.md
index 041756c..50e0786 100644
--- a/README.md
+++ b/README.md
@@ -1,54 +1,61 @@
+# README.md
 # OceansGuard
 
-OceansGuard は、生成AIによるコード変更を  
-**CI・契約・セキュリティで機械的に裁くためのガードレール**です。
-
-## 目的
-- AIにコードを書かせても事故らせない
-- 人が説明・確認・判断しなくてよい開発
-- どの言語・フレームワークでも共通運用
-
-## 基本思想
-- AIは「提案者」
-- 正しさは「テスト・契約・ポリシー」が決める
-- 通らない変更は採用されない
-
-## 使い方（各プロジェクト側）
-```bash
-python path/to/aiguard.py init
-python path/to/aiguard.py pack
-python path/to/aiguard.py check
-
-対応フェーズ
-
-開発前 / 開発途中 / 開発後 すべて対応
-
-
----
-
-## ③ あなたの「不可がほぼ無い」運用フロー（確定）
-**どの案件でもこれだけ**
-
-
-
-AIに投げる前 → ai:pack
-AI差分適用後 → ai:check
-通ったら → 採用
-
-
-- 考えない
-- 説明しない
-- レビューしない  
-
----
-
-## ④ 最初のGit操作（推奨）
-```bash
-git add .
-git commit -m "feat: initial OceansGuard core structure"
-git tag v0.1.0
-git push origin main --tags
-
-
-
-## Create by OceansCreative
\ No newline at end of file
+AI-assisted development guardrails for any repository.
+
+## What it solves
+- AI-generated changes that accidentally drop existing code
+- Lack of global context (only partial files shown)
+- Forgetfulness / inconsistent constraints across sessions
+- No test / lint guarantees
+- Secret leakage (keys/tokens) into commits
+- Risky full-rewrite changes
+
+## Core commands
+
+### init
+Create minimal guard files in target repo (idempotent; no overwrite).
+
+python core/aiguard.py init --repo .
+
+### pack
+Generate AI context pack (diff-first).
+```
+python core/aiguard.py pack --repo .
+```
+### check
+Run guard checks + configured project checks and write reports.
+```
+python core/aiguard.py check --repo .
+```
+### run
+Shortcut = pack + check.
+```
+python core/aiguard.py run --repo . --task "your task"
+```
+## Strict mode
+--strict makes guardrails non-negotiable:
+- requires PyYAML
+- fails if checks.commands is empty
+- fails if contracts/openapi.json is missing/empty
+```
+python core/aiguard.py run --repo . --task "CI guard" --strict
+```
+
+## Submodule usage (recommended)
+In your target repository:
+```
+git submodule add https://github.com/OceansCreative/OceansGuard.git tools/OceansGuard
+python tools/OceansGuard/core/aiguard.py init --repo .
+python tools/OceansGuard/core/aiguard.py run --repo . --task "初回ガード適用"
+```
+## Outputs
+- ai_context_pack.md: single file to paste into AI chat
+- ai_test_last.log: raw execution logs
+- ai_check_report.json: structured result for CI/PR gating
+
+## Git hooks (prevent committing to main)
+Install with:
+```
+python core/install_hooks.py --repo .
+```
\ No newline at end of file
diff --git a/core/aiguard.py b/core/aiguard.py
index dfad531..608d5df 100644
--- a/core/aiguard.py
+++ b/core/aiguard.py
@@ -2,156 +2,286 @@
 from __future__ import annotations
 
 import argparse
+import fnmatch
+import hashlib
 import json
 import os
-import shutil
+import re
 import subprocess
-import sys
 from dataclasses import dataclass
 from datetime import datetime
 from pathlib import Path
-from typing import Any
-
-
-PROJECT_NAME = "OceansGuard"
+from typing import Any, Dict, List, Optional, Tuple
 
 
+# =========================
+# Utilities
+# =========================
 def now_iso() -> str:
     return datetime.now().isoformat(timespec="seconds")
 
 
 def info(msg: str) -> None:
-    print(f"[{PROJECT_NAME}] {msg}")
+    print(f"[OceansGuard] {msg}")
 
 
 def warn(msg: str) -> None:
-    print(f"[{PROJECT_NAME}][WARN] {msg}")
+    print(f"[OceansGuard][WARN] {msg}")
 
 
 def die(msg: str, code: int = 1) -> None:
-    raise SystemExit(f"[{PROJECT_NAME}] {msg}")
+    raise SystemExit(f"[OceansGuard] {msg}")
 
 
-def copy_if_missing(src: Path, dst: Path) -> None:
-    if dst.exists():
-        info(f"[skip] exists: {dst}")
-        return
-    dst.parent.mkdir(parents=True, exist_ok=True)
-    shutil.copyfile(src, dst)
-    info(f"[create] {dst}")
+def sha256_text(s: str) -> str:
+    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
 
 
-def write_if_missing(dst: Path, content: str) -> None:
-    if dst.exists():
-        info(f"[skip] exists: {dst}")
-        return
-    dst.parent.mkdir(parents=True, exist_ok=True)
-    dst.write_text(content, encoding="utf-8")
-    info(f"[create] {dst}")
+def run_argv(argv, cwd=None):
+    import subprocess, os
 
+    p = subprocess.run(
+        argv,
+        cwd=cwd,
+        capture_output=True,
+        env=os.environ.copy(),
+    )
 
-def read_text_if_exists(p: Path) -> str | None:
-    if not p.exists():
-        return None
-    try:
-        return p.read_text(encoding="utf-8", errors="ignore")
-    except OSError:
-        return None
+    def _decode(b: bytes) -> str:
+        if not b:
+            return ""
+        try:
+            return b.decode("utf-8")
+        except UnicodeDecodeError:
+            return b.decode("utf-8", errors="replace")
 
+    p.stdout = _decode(p.stdout)
+    p.stderr = _decode(p.stderr)
+    return p
 
-def safe_json_load(p: Path) -> Any | None:
+
+
+def run_shell(cmd, cwd=None):
+    import subprocess, os
+
+    p = subprocess.run(
+        cmd,
+        cwd=cwd,
+        shell=True,
+        capture_output=True,
+        env=os.environ.copy(),
+    )
+
+    def _decode(b: bytes) -> str:
+        if not b:
+            return ""
+        try:
+            return b.decode("utf-8")
+        except UnicodeDecodeError:
+            return b.decode("utf-8", errors="replace")
+
+    p.stdout = _decode(p.stdout)
+    p.stderr = _decode(p.stderr)
+    return p
+
+
+def safe_read_text(p: Path, max_kb: int) -> str:
     try:
-        return json.loads(p.read_text(encoding="utf-8"))
-    except Exception:
-        return None
+        b = p.read_bytes()
+    except Exception as e:
+        return f"(failed to read: {e})\n"
+    if len(b) > max_kb * 1024:
+        return f"(skipped: too large {len(b)} bytes > {max_kb}KB)\n"
+    return b.decode("utf-8", errors="replace")
+
 
+def guard_root() -> Path:
+    # core/aiguard.py → OceansGuard/
+    return Path(__file__).resolve().parent.parent
 
-def try_import_yaml():
+
+def ensure_pyyaml(strict: bool) -> bool:
     try:
-        import yaml  # type: ignore
-        return yaml
+        import yaml  # noqa: F401
+        return True
     except Exception:
-        return None
+        if strict:
+            die("PyYAML is required in --strict mode. Install: pip install pyyaml")
+        warn("PyYAML not found. Some config-driven features may be skipped.")
+        return False
 
 
-@dataclass(frozen=True)
-class GuardOutput:
-    pack: str = "ai_context_pack.md"
-    audit: str = "CHANGELOG_AI.md"
-    testlog: str = "ai_test_last.log"
-
-
-@dataclass(frozen=True)
-class GuardConfig:
-    raw: dict[str, Any]
-    output: GuardOutput
-
-    @staticmethod
-    def load(repo: Path) -> "GuardConfig":
-        """
-        優先順位:
-        1) repo/.aiguard.yml
-        2) templates/.aiguard.yml を init がコピー済みならそれ
-        3) templates/.aiguard.yml を repo にコピーしてから読む
-        4) 最終的に空設定（最低限で通す）
-        """
-        cfg_path = repo / ".aiguard.yml"
-        if not cfg_path.exists():
-            # まず templates が同リポ内にある前提（OceansGuard自身）
-            # 他PJで submodule 利用のケースでも templates が来る想定
-            tpl = repo / "templates" / ".aiguard.yml"
-            if tpl.exists():
-                copy_if_missing(tpl, cfg_path)
-
-        if not cfg_path.exists():
-            warn(".aiguard.yml not found. Running with minimal defaults.")
-            raw = {}
-            return GuardConfig(raw=raw, output=GuardOutput())
-
-        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
-        yaml = try_import_yaml()
-        if yaml is None:
-            warn("PyYAML not installed. Some YAML features may not be parsed. "
-                 "Install: pip install pyyaml (CI already best-effort installs it).")
-            # 最低限: JSONとして読めるなら読む、無理なら空
-            raw = {}
-            return GuardConfig(raw=raw, output=GuardOutput())
+# =========================
+# Config (.aiguard.yml)
+# =========================
+@dataclass
+class ContextSmall:
+    include: List[str]
+
+
+@dataclass
+class ContextLarge:
+    roots: List[str]
+    exclude_dirs: List[str]
+    exclude_globs: List[str]
+    max_files: int
+    max_kb_each: int
+
+
+@dataclass
+class Evidence:
+    commands: List[str]
+
+
+@dataclass
+class Dlp:
+    enable: bool
+    block_on_detect: bool
+    mask: bool
+    allowlist_files: List[str]
+
+
+@dataclass
+class Guard:
+    forbid_full_rewrite: bool
+    allow_full_rewrite_globs: List[str]
+
+
+@dataclass
+class Checks:
+    commands: List[str]
+
+
+@dataclass
+class Output:
+    pack: str
+    audit: str
+    testlog: str
+    report_json: str
+
+
+@dataclass
+class Config:
+    version: int
+    context_small: ContextSmall
+    context_large: ContextLarge
+    evidence: Evidence
+    dlp: Dlp
+    guard: Guard
+    checks: Checks
+    output: Output
+
 
+def _dict_get(d: dict, key: str, default):
+    v = d.get(key, default)
+    return default if v is None else v
+
+
+def load_config(repo: Path, strict: bool) -> Config:
+    has_yaml = ensure_pyyaml(strict=strict)
+    cfg_path = repo / ".aiguard.yml"
+    if not cfg_path.exists():
+        die(".aiguard.yml not found. Run init first.")
+
+    raw: Dict[str, Any] = {}
+    if has_yaml:
         try:
-            raw = yaml.safe_load(text) or {}
+            import yaml  # type: ignore
+            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
         except Exception as e:
-            warn(f"Failed to parse .aiguard.yml: {e}")
+            if strict:
+                die(f"Failed to parse .aiguard.yml: {e}")
+            warn(f"Failed to parse .aiguard.yml; using minimal defaults. ({e})")
             raw = {}
+    else:
+        if strict:
+            die("Cannot read .aiguard.yml without PyYAML in strict mode.")
+
+    version = int(_dict_get(raw, "version", 1))
+
+    ctx = _dict_get(raw, "context", {})
+    small = _dict_get(ctx, "small", {})
+    large = _dict_get(ctx, "large", {})
+
+    context_small = ContextSmall(include=[str(x) for x in _dict_get(small, "include", [])])
+    context_large = ContextLarge(
+        roots=[str(x) for x in _dict_get(large, "roots", ["backend", "app", "src", "frontend"])],
+        exclude_dirs=[str(x) for x in _dict_get(large, "exclude_dirs", [
+            ".git", ".venv", "venv", "node_modules", "dist", "build", ".next",
+            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
+        ])],
+        exclude_globs=[str(x) for x in _dict_get(large, "exclude_globs", ["**/*.min.js", "**/*.map"])],
+        max_files=int(_dict_get(large, "max_files", 220)),
+        max_kb_each=int(_dict_get(large, "max_kb_each", 64)),
+    )
+
+    ev = _dict_get(raw, "evidence", {})
+    evidence = Evidence(commands=[str(x) for x in _dict_get(ev, "commands", [])])
+
+    dlp_raw = _dict_get(raw, "dlp", {})
+    dlp = Dlp(
+        enable=bool(_dict_get(dlp_raw, "enable", True)),
+        block_on_detect=bool(_dict_get(dlp_raw, "block_on_detect", True)),
+        mask=bool(_dict_get(dlp_raw, "mask", True)),
+        allowlist_files=[str(x) for x in _dict_get(dlp_raw, "allowlist_files", [])],
+    )
+
+    guard_raw = _dict_get(raw, "guard", {})
+    guard = Guard(
+        forbid_full_rewrite=bool(_dict_get(guard_raw, "forbid_full_rewrite", True)),
+        allow_full_rewrite_globs=[str(x) for x in _dict_get(guard_raw, "allow_full_rewrite_globs", [])],
+    )
+
+    checks_raw = _dict_get(raw, "checks", {})
+    checks = Checks(commands=[str(x) for x in _dict_get(checks_raw, "commands", [])])
+
+    out_raw = _dict_get(raw, "output", {})
+    output = Output(
+        pack=str(_dict_get(out_raw, "pack", "ai_context_pack.md")),
+        audit=str(_dict_get(out_raw, "audit", "CHANGELOG_AI.md")),
+        testlog=str(_dict_get(out_raw, "testlog", "ai_test_last.log")),
+        report_json=str(_dict_get(out_raw, "report_json", "ai_check_report.json")),
+    )
+
+    return Config(
+        version=version,
+        context_small=context_small,
+        context_large=context_large,
+        evidence=evidence,
+        dlp=dlp,
+        guard=guard,
+        checks=checks,
+        output=output,
+    )
+
+
+# =========================
+# init
+# =========================
+def copy_if_missing(src: Path, dst: Path) -> None:
+    if dst.exists():
+        info(f"skip exists: {dst}")
+        return
+    dst.parent.mkdir(parents=True, exist_ok=True)
+    dst.write_bytes(src.read_bytes())
+    info(f"create: {dst}")
+
+
+def write_if_missing(dst: Path, content: str) -> None:
+    if dst.exists():
+        info(f"skip exists: {dst}")
+        return
+    dst.parent.mkdir(parents=True, exist_ok=True)
+    dst.write_text(content, encoding="utf-8")
+    info(f"create: {dst}")
+
+
+def cmd_init(repo: Path) -> None:
+    tpl = guard_root() / "templates" / ".aiguard.yml"
+    if not tpl.exists():
+        die("templates/.aiguard.yml not found in OceansGuard")
+    copy_if_missing(tpl, repo / ".aiguard.yml")
 
-        out = raw.get("output") or {}
-        output = GuardOutput(
-            pack=str(out.get("pack", GuardOutput.pack)),
-            audit=str(out.get("audit", GuardOutput.audit)),
-            testlog=str(out.get("testlog", GuardOutput.testlog)),
-        )
-        return GuardConfig(raw=raw, output=output)
-
-
-def cmd_init(repo: Path) -> int:
-    """
-    init は「コピー専用」思想を維持。
-    - templates/.aiguard.yml を .aiguard.yml にコピー（上書きしない）
-    - ARCHITECTURE.md / SNAPSHOT.md を無ければ作成
-    - contracts/ を無ければ作成
-    - contracts/openapi.json が無ければ placeholder を作成（上書きしない）
-    """
-    here = Path(__file__).resolve()
-    guard_root = here.parent.parent  # OceansGuard/（このリポのルート想定）
-    templates = guard_root / "templates"
-
-    tpl_aiguard = templates / ".aiguard.yml"
-    if not tpl_aiguard.exists():
-        die("templates/.aiguard.yml not found")
-
-    # 1) .aiguard.yml
-    copy_if_missing(tpl_aiguard, repo / ".aiguard.yml")
-
-    # 2) ARCHITECTURE.md / SNAPSHOT.md
     write_if_missing(
         repo / "ARCHITECTURE.md",
         "# ARCHITECTURE\n\n"
@@ -169,229 +299,407 @@ def cmd_init(repo: Path) -> int:
         f"_generated by OceansGuard init @ {now_iso()}_\n",
     )
 
-    # 3) contracts/
     contracts = repo / "contracts"
-    if not contracts.exists():
-        contracts.mkdir(parents=True, exist_ok=True)
-        (contracts / "README.md").write_text(
-            "# Contracts\n\n"
-            "このディレクトリには、守るべき契約を置きます。\n\n"
-            "- OpenAPI（FastAPI の openapi.yaml / openapi.json）\n"
-            "- DB schema snapshot\n"
-            "- UI DTO / 型\n\n",
-            encoding="utf-8",
-        )
-        info(f"[create] {contracts}/README.md")
-    else:
-        info(f"[skip] exists: {contracts}/")
-
-    # 4) contracts/openapi.json placeholder（空ファイル対策：ただし上書きしない）
-    openapi = contracts / "openapi.json"
-    if not openapi.exists():
-        openapi.write_text(
-            json.dumps(
-                {
-                    "openapi": "3.0.3",
-                    "info": {"title": f"{PROJECT_NAME} Placeholder API", "version": "0.0.0"},
-                    "paths": {},
-                },
-                ensure_ascii=False,
-                indent=2,
-            ) + "\n",
-            encoding="utf-8",
-        )
-        info(f"[create] {openapi}")
+    contracts.mkdir(parents=True, exist_ok=True)
+    write_if_missing(
+        contracts / "README.md",
+        "# Contracts\n\n"
+        "このディレクトリには、守るべき契約（スキーマ/仕様）を置きます。\n\n"
+        "- OpenAPI: contracts/openapi.json（または openapi.yaml）\n"
+        "- DB schema snapshot\n"
+        "- DTO/型\n",
+    )
+    if not (contracts / "openapi.json").exists():
+        (contracts / "openapi.json").write_text("", encoding="utf-8")
+        info("create: contracts/openapi.json (empty)")
+
+    info("init completed")
+
+
+# =========================
+# pack (diff-first)
+# =========================
+def pack_git_section(repo: Path) -> str:
+    buf: List[str] = []
+    buf.append("## Git\n")
+
+    st = run_argv(["git", "status", "--porcelain=v1"], cwd=repo)
+    buf.append("\n### git status --porcelain=v1\n```text\n")
+    buf.append(st.stdout or "")
+    buf.append("\n```\n")
+
+    df = run_argv(["git", "diff"], cwd=repo)
+    diff_text = df.stdout or ""
+    buf.append("\n### git diff\n```diff\n")
+    if len(diff_text) > 200_000:
+        buf.append(diff_text[:200_000])
+        buf.append("\n... (truncated)\n")
     else:
-        info(f"[skip] exists: {openapi}")
+        buf.append(diff_text)
+    buf.append("\n```\n")
 
-    info("[OK] init completed")
-    return 0
+    return "".join(buf)
 
 
-def _ensure_guard_dir(repo: Path) -> Path:
-    d = repo / ".aiguard"
-    d.mkdir(parents=True, exist_ok=True)
-    return d
+def glob_files(root: Path, pattern: str) -> List[Path]:
+    return [p for p in root.glob(pattern) if p.is_file()]
 
 
-def _collect_context_pack(repo: Path, cfg: GuardConfig) -> str:
-    """
-    templates/.aiguard.yml の context.small.include を中心に “軽量” にまとめる。
-    無い場合は、代表ファイルだけを対象にする。
-    """
-    raw = cfg.raw
-    includes: list[str] = []
+def should_exclude(path: Path, repo: Path, exclude_dirs: List[str], exclude_globs: List[str]) -> bool:
+    rel = path.relative_to(repo).as_posix()
+    parts = set(Path(rel).parts)
+    if any(d in parts for d in exclude_dirs):
+        return True
+    for g in exclude_globs:
+        if fnmatch.fnmatch(rel, g):
+            return True
+    return False
 
-    ctx = raw.get("context") or {}
-    small = ctx.get("small") or {}
-    include = small.get("include")
-    if isinstance(include, list):
-        includes = [str(x) for x in include]
-    else:
-        includes = ["ARCHITECTURE.md", "SNAPSHOT.md", "README.md", "pyproject.toml", "package.json", "contracts/**"]
-
-    lines: list[str] = []
-    lines.append(f"# {PROJECT_NAME} Context Pack")
-    lines.append("")
-    lines.append(f"_generated @ {now_iso()}_")
-    lines.append("")
-
-    def add_file(path: Path) -> None:
-        rel = path.relative_to(repo)
-        text = read_text_if_exists(path)
-        if text is None:
-            return
-        # 過大防止：1ファイル最大 4000 文字
-        text = text[:4000]
-        lines.append(f"## {rel.as_posix()}")
-        lines.append("```")
-        lines.append(text.rstrip("\n"))
-        lines.append("```")
-        lines.append("")
-
-    for pat in includes:
-        if pat.endswith("/**") or pat.endswith("**"):
-            base = pat.replace("/**", "").replace("**", "").strip("/")
-            base_dir = repo / base
-            if base_dir.exists() and base_dir.is_dir():
-                for p in sorted(base_dir.rglob("*")):
-                    if p.is_file() and p.suffix.lower() in (".md", ".yml", ".yaml", ".json", ".toml", ".py", ".ts", ".tsx", ".js"):
-                        add_file(p)
-            continue
 
-        # glob対応
-        matched = list(repo.glob(pat))
-        if matched:
-            for p in matched:
-                if p.is_file():
-                    add_file(p)
+def collect_small(repo: Path, cfg: Config) -> List[Path]:
+    out: List[Path] = []
+    for pat in cfg.context_small.include:
+        out.extend(glob_files(repo, pat))
+    seen = set()
+    uniq: List[Path] = []
+    for p in sorted(out, key=lambda x: x.relative_to(repo).as_posix()):
+        rel = p.relative_to(repo).as_posix()
+        if rel in seen:
             continue
+        seen.add(rel)
+        uniq.append(p)
+    return uniq
 
-        # 通常ファイル
-        p = repo / pat
+
+def collect_changed_files(repo: Path) -> List[Path]:
+    cp = run_argv(["git", "diff", "--name-only"], cwd=repo)
+    files = []
+    for ln in (cp.stdout or "").splitlines():
+        ln = ln.strip()
+        if not ln:
+            continue
+        p = repo / ln
         if p.exists() and p.is_file():
-            add_file(p)
+            files.append(p)
+    return files
+
+
+def collect_large(repo: Path, cfg: Config) -> List[Path]:
+    files: List[Path] = []
+    roots = cfg.context_large.roots[:] if cfg.context_large.roots else ["."]
+    for r in roots:
+        base = (repo / r).resolve()
+        if not base.exists():
+            continue
+        for p in base.rglob("*"):
+            if not p.is_file():
+                continue
+            if should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs):
+                continue
+            files.append(p)
+    files = sorted(files, key=lambda x: x.relative_to(repo).as_posix())
+    if len(files) > cfg.context_large.max_files:
+        files = files[: cfg.context_large.max_files]
+    return files
+
+
+def evidence_section(repo: Path, cfg: Config) -> str:
+    buf: List[str] = []
+    buf.append("## Evidence\n")
+    for cmd in cfg.evidence.commands:
+        buf.append(f"\n### $ {cmd}\n")
+        cp = run_shell(cmd, cwd=repo)
+        buf.append("```text\n")
+        buf.append(cp.stdout or "")
+        buf.append("\n```\n")
+    return "".join(buf)
+
+
+def pack_files(repo: Path, files: List[Path], max_kb_each: int, title: str) -> str:
+    buf: List[str] = []
+    buf.append(f"## {title}\n")
+    buf.append("\n### File list\n```text\n")
+    for p in files:
+        buf.append(p.relative_to(repo).as_posix() + "\n")
+    buf.append("```\n\n")
+
+    for p in files:
+        rel = p.relative_to(repo).as_posix()
+        buf.append(f"### {rel}\n")
+        buf.append("```text\n")
+        buf.append(safe_read_text(p, max_kb=max_kb_each))
+        buf.append("\n```\n\n")
+    return "".join(buf)
+
+
+def cmd_pack(repo: Path, strict: bool) -> None:
+    cfg = load_config(repo, strict=strict)
+    out_path = repo / cfg.output.pack
+
+    small_files = collect_small(repo, cfg)
+
+    # diff-first: 変更ファイルを優先的に添付（除外規則も適用）
+    changed = [
+        p for p in collect_changed_files(repo)
+        if not should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs)
+    ]
+    # large は補助（上限付き）
+    large_files = collect_large(repo, cfg)
+
+    buf: List[str] = []
+    buf.append("# OceansGuard Context Pack\n\n")
+    buf.append(f"- generated_at: {now_iso()}\n")
+    buf.append(f"- repo: {repo}\n")
+    buf.append(f"- config_version: {cfg.version}\n")
+    buf.append(f"- mode: {'strict' if strict else 'normal'}\n\n")
+
+    buf.append(pack_git_section(repo))
+    buf.append(evidence_section(repo, cfg))
+
+    if changed:
+        buf.append(pack_files(repo, changed, max_kb_each=cfg.context_large.max_kb_each, title="Changed files (diff-first)"))
+    if small_files:
+        buf.append(pack_files(repo, small_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (small)"))
+    if large_files:
+        buf.append(pack_files(repo, large_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (large, capped)"))
+
+    out_path.write_text("".join(buf), encoding="utf-8")
+    info(f"pack written: {out_path}")
+
+
+# =========================
+# Guard: full rewrite detection
+# =========================
+def detect_full_rewrite(repo: Path, forbid: bool, allow_globs: List[str]) -> Optional[str]:
+    if not forbid:
+        return None
+
+    cp = run_argv(["git", "diff", "--numstat"], cwd=repo)
+    suspicious: List[str] = []
+
+    def _allowed(path_str: str) -> bool:
+        # numstat の file は基本 posix だが、念のため正規化
+        s = path_str.replace("\\", "/")
+        for g in allow_globs or []:
+            if fnmatch.fnmatch(s, g):
+                return True
+        return False
+
+    for ln in (cp.stdout or "").splitlines():
+        parts = ln.split("\t")
+        if len(parts) != 3:
+            continue
+        add_s, del_s, file_s = parts
+        if add_s == "-" or del_s == "-":
+            continue
+
+        if _allowed(file_s):
+            continue
+
+        try:
+            add_n = int(add_s)
+            del_n = int(del_s)
+        except ValueError:
+            continue
+
+        if (add_n + del_n) >= 800 and add_n >= 300 and del_n >= 300:
+            suspicious.append(f"{file_s} (add={add_n}, del={del_n})")
+
+    if suspicious:
+        return "Possible full rewrite detected:\n" + "\n".join(f"- {s}" for s in suspicious)
+    return None
+
+
+
+# =========================
+# DLP
+# =========================
+SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
+    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
+    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
+    ("GITHUB_TOKEN", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
+    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
+    ("JWT_LIKE", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
+    ("PASSWORD_ASSIGN", re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]")),
+]
+
+
+def is_allowlisted(rel_posix: str, allowlist: List[str]) -> bool:
+    for pat in allowlist:
+        if fnmatch.fnmatch(rel_posix, pat):
+            return True
+    return False
 
-    return "\n".join(lines).rstrip() + "\n"
 
+def scan_dlp(repo: Path, enable: bool, allowlist: List[str], mask: bool, strict: bool) -> List[str]:
+    if not enable:
+        return []
+    max_bytes = 256 * 1024  # 256KB
 
-def cmd_pack(repo: Path) -> int:
-    """
-    pack:
-    - .aiguard/ai_context_pack.md を生成（output.pack に従う）
-    - .aiguard/pack.json に簡易メタを書き出し
-    - ここでは “止めない” 方針（構成生成が目的）
-    """
-    cfg = GuardConfig.load(repo)
-    guard_dir = _ensure_guard_dir(repo)
+    cp = run_argv(["git", "ls-files"], cwd=repo)
+    hits: List[str] = []
 
-    pack_md = _collect_context_pack(repo, cfg)
-    (repo / cfg.output.pack).write_text(pack_md, encoding="utf-8")
-    info(f"[write] {cfg.output.pack}")
+    for rel in (cp.stdout or "").splitlines():
+        rel = rel.strip()
+        if not rel:
+            continue
+        if is_allowlisted(rel, allowlist):
+            continue
+        p = repo / rel
+        if not p.exists() or not p.is_file():
+            continue
+        try:
+            b = p.read_bytes()
+        except Exception:
+            continue
+        if len(b) > max_bytes:
+            continue
+        s = b.decode("utf-8", errors="ignore")
+        for name, pat in SECRET_PATTERNS:
+            m = pat.search(s)
+            if not m:
+                continue
+            sample = m.group(0)
+            if mask:
+                sample = f"<masked:{sha256_text(sample)[:12]}>"
+            hits.append(f"{name}: {rel}: {sample}")
+
+    # strict は「検知ルールのスキップ」ではなく、検知をより厳密にする余地（今は同等）
+    return hits
+
+
+# =========================
+# OpenAPI contract
+# =========================
+def openapi_contract_check(repo: Path, strict: bool) -> Optional[str]:
+    p = repo / "contracts" / "openapi.json"
+    if not p.exists():
+        if strict:
+            return "contracts/openapi.json is required in strict mode (can be empty only in normal mode)"
+        return None
 
-    meta = {
-        "project": PROJECT_NAME,
+    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
+    if not txt:
+        if strict:
+            return "contracts/openapi.json is empty in strict mode"
+        return None
+
+    try:
+        obj = json.loads(txt)
+    except Exception as e:
+        return f"contracts/openapi.json is not valid JSON: {e}"
+
+    if not isinstance(obj, dict):
+        return "contracts/openapi.json must be a JSON object"
+    if "openapi" not in obj and "swagger" not in obj:
+        return "contracts/openapi.json missing 'openapi' (or 'swagger') field"
+    return None
+
+
+# =========================
+# check (report json)
+# =========================
+def cmd_check(repo: Path, strict: bool) -> None:
+    cfg = load_config(repo, strict=strict)
+
+    report: Dict[str, Any] = {
+        "tool": "OceansGuard",
         "generated_at": now_iso(),
-        "output": dataclass_to_dict(cfg.output),
-        "note": "pack generates context artifacts. It should be non-blocking.",
+        "repo": str(repo),
+        "mode": "strict" if strict else "normal",
+        "checks": [],
+        "dlp_hits": [],
+        "guard": {"full_rewrite": None},
+        "openapi_contract": None,
+        "status": "pass",
     }
-    (guard_dir / "pack.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
-    info("[OK] pack completed")
-    return 0
-
 
-def _run_command(cmd: str, repo: Path) -> tuple[int, str]:
-    """
-    YAMLの checks.commands は shell 前提のものがあるため shell=True で実行。
-    """
-    p = subprocess.Popen(
-        cmd,
-        cwd=str(repo),
-        shell=True,
-        stdout=subprocess.PIPE,
-        stderr=subprocess.STDOUT,
-        text=True,
+    # 1) full rewrite guard
+    fr = detect_full_rewrite(repo, forbid=cfg.guard.forbid_full_rewrite, allow_globs=cfg.guard.allow_full_rewrite_globs)
+
+    if fr:
+        report["guard"]["full_rewrite"] = fr
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die(fr)
+
+    # 2) DLP
+    dlp_hits = scan_dlp(
+        repo,
+        enable=cfg.dlp.enable,
+        allowlist=cfg.dlp.allowlist_files,
+        mask=cfg.dlp.mask,
+        strict=strict,
     )
-    out, _ = p.communicate()
-    return p.returncode or 0, out or ""
-
-
-def _load_checks_commands(cfg: GuardConfig) -> list[str]:
-    raw = cfg.raw
-    checks = raw.get("checks") or {}
-    commands = checks.get("commands")
-    if isinstance(commands, list):
-        return [str(x) for x in commands]
-    return []
-
-
-def cmd_check(repo: Path) -> int:
-    """
-    check:
-    - templates/.aiguard.yml にある checks.commands を順に実行
-    - 失敗があれば exit 1（＝PRを止める）
-    - ただし summarize は || true が付いている想定なので失敗しても継続
-    """
-    cfg = GuardConfig.load(repo)
-    guard_dir = _ensure_guard_dir(repo)
-
-    cmds = _load_checks_commands(cfg)
-    if not cmds:
-        warn("No checks.commands found in .aiguard.yml. Nothing to run.")
-        # 何も無いなら成功扱い（ガードレールとしては最小）
-        return 0
-
-    log_lines: list[str] = []
-    log_lines.append(f"[{PROJECT_NAME}] check started @ {now_iso()}")
-    log_lines.append("")
-
-    failed = False
-
-    for i, cmd in enumerate(cmds, start=1):
-        log_lines.append(f"---")
-        log_lines.append(f"[{i}/{len(cmds)}] $ {cmd}")
-        rc, out = _run_command(cmd, repo)
-        log_lines.append(out.rstrip("\n"))
-        log_lines.append(f"[exit] {rc}")
-        log_lines.append("")
-
-        # shellで '|| true' を付けているものは rc=0 になる想定
-        if rc != 0:
-            failed = True
-
-    testlog_path = repo / cfg.output.testlog
-    testlog_path.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")
-    info(f"[write] {cfg.output.testlog}")
-
-    # 失敗ログは guard_dir にもコピーしておく（CIで参照しやすい）
-    (guard_dir / "ai_test_last.log").write_text(testlog_path.read_text(encoding="utf-8"), encoding="utf-8")
+    report["dlp_hits"] = dlp_hits
+    if dlp_hits and cfg.dlp.block_on_detect:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die("DLP detected potential secrets:\n" + "\n".join(f"- {h}" for h in dlp_hits))
+
+    # 3) OpenAPI contract
+    oc = openapi_contract_check(repo, strict=strict)
+    report["openapi_contract"] = oc
+    if oc:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die(oc)
+
+    # 4) checks.commands
+    log_path = repo / cfg.output.testlog
+    logs: List[str] = []
+    logs.append(f"[OceansGuard] check started @ {now_iso()}\n")
+
+    failed: List[Tuple[str, int]] = []
+
+    for i, cmd in enumerate(cfg.checks.commands, start=1):
+        cmd = (cmd or "").strip()
+        if not cmd:
+            continue
+        info(f"check[{i}] {cmd}")
+        cp = run_shell(cmd, cwd=repo)
+        entry = {
+            "index": i,
+            "command": cmd,
+            "exit_code": cp.returncode,
+        }
+        report["checks"].append(entry)
+
+        logs.append(f"\n=== check[{i}] {cmd} ===\n")
+        logs.append(cp.stdout or "")
+        logs.append(f"\n[exit_code] {cp.returncode}\n")
+
+        if cp.returncode != 0:
+            failed.append((cmd, cp.returncode))
+
+    log_path.write_text("".join(logs), encoding="utf-8")
+    info(f"testlog written: {log_path}")
+
+    # strict: checks.commands が空なら fail（「テスト保証」を強制）
+    if strict and not cfg.checks.commands:
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        die("checks.commands is empty in strict mode")
 
     if failed:
-        die("check failed (one or more commands returned non-zero).", code=1)
-    info("[OK] check completed")
-    return 0
-
+        report["status"] = "fail"
+        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+        summary = "\n".join([f"- ({rc}) {cmd}" for cmd, rc in failed])
+        die("check failed:\n" + summary)
 
-def cmd_run(repo: Path, task: str) -> int:
-    """
-    将来拡張用。現時点では最低限として、
-    - task が空なら check を実行
-    - task が init/pack/check のいずれかならそのまま実行
-    """
-    if not task:
-        return cmd_check(repo)
-    if task == "init":
-        return cmd_init(repo)
-    if task == "pack":
-        return cmd_pack(repo)
-    if task == "check":
-        return cmd_check(repo)
-    warn(f"Unknown task: {task}. Running check instead.")
-    return cmd_check(repo)
+    (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
+    info(f"report written: {repo / cfg.output.report_json}")
+    info("check passed")
 
 
-def dataclass_to_dict(obj: Any) -> dict[str, Any]:
-    return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
+# =========================
+# run
+# =========================
+def cmd_run(repo: Path, task: str, strict: bool) -> None:
+    if task.strip():
+        info(f"run task: {task}")
+    cmd_pack(repo, strict=strict)
+    cmd_check(repo, strict=strict)
 
 
 def main() -> None:
@@ -399,20 +707,25 @@ def main() -> None:
     ap.add_argument("command", choices=["init", "pack", "check", "run"])
     ap.add_argument("--repo", default=".")
     ap.add_argument("--task", default="")
+    ap.add_argument("--strict", action="store_true", help="Do not allow skipping; fail if prerequisites missing.")
     args = ap.parse_args()
 
     repo = Path(args.repo).resolve()
 
     if args.command == "init":
-        raise SystemExit(cmd_init(repo))
+        cmd_init(repo)
+        return
     if args.command == "pack":
-        raise SystemExit(cmd_pack(repo))
+        cmd_pack(repo, strict=args.strict)
+        return
     if args.command == "check":
-        raise SystemExit(cmd_check(repo))
+        cmd_check(repo, strict=args.strict)
+        return
     if args.command == "run":
-        raise SystemExit(cmd_run(repo, args.task))
+        cmd_run(repo, task=args.task, strict=args.strict)
+        return
 
-    raise SystemExit(0)
+    die("unknown command")
 
 
 if __name__ == "__main__":
diff --git a/templates/.aiguard.yml b/templates/.aiguard.yml
index 6b1bdcc..a8dbaaf 100644
--- a/templates/.aiguard.yml
+++ b/templates/.aiguard.yml
@@ -21,19 +21,21 @@ context:
       - build
       - .next
       - __pycache__
+      - __pypackages__
       - .pytest_cache
       - .mypy_cache
       - .ruff_cache
     exclude_globs: ["**/*.min.js", "**/*.map"]
     max_files: 220
     max_kb_each: 64
-  evidence:
-    commands:
-      - git status --porcelain=v1 || true
-      - git diff || true
-      - python --version || true
-      - node --version || true
-      - npm --version || true
+
+evidence:
+  commands:
+    - git status --porcelain=v1 || true
+    - git diff || true
+    - python --version || true
+    - node --version || true
+    - npm --version || true
 
 dlp:
   enable: true
@@ -44,37 +46,33 @@ dlp:
     - "frontend/.env.example"
 
 guard:
-  max_files: 10
-  max_lines: 400
   forbid_full_rewrite: true
+  allow_full_rewrite_globs:
+    - "core/aiguard.py"
+    - ".github/workflows/oceansguard.yml"
 
 checks:
   commands:
     # =========================
-    # Backend (FastAPI / Python)
+    # Backend (Python)
     # =========================
-
-    # compileall: backend/app/src が存在する場合のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       sys.exit(subprocess.call([sys.executable,'-m','compileall',*targets]) if targets else 0)"
 
-    # ruff: インストール済み かつ 対象dir存在時のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       has=importlib.util.find_spec('ruff') is not None;
       sys.exit(subprocess.call([sys.executable,'-m','ruff','check',*targets]) if (has and targets) else 0)"
 
-    # mypy: インストール済み かつ 対象dir存在時のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
       has=importlib.util.find_spec('mypy') is not None;
       sys.exit(subprocess.call([sys.executable,'-m','mypy',*targets]) if (has and targets) else 0)"
 
-    # pytest: tests が存在し、pytest導入済みの場合のみ実行（無ければスキップ）
     - >
       python -c "import os,sys,subprocess,importlib.util;
       has=importlib.util.find_spec('pytest') is not None;
@@ -82,34 +80,20 @@ checks:
       sys.exit(subprocess.call([sys.executable,'-m','pytest','-q']) if (has and has_tests) else 0)"
 
     # =========================
-    # Frontend (React / Node)
+    # Frontend (React)
     # =========================
-
-    # npm ci (or npm install): frontend が存在する場合のみ実行
     - >
       python -c "import os,sys,subprocess;
       d='frontend';
       sys.exit(0 if not os.path.isdir(d) else (subprocess.call('npm ci --silent', cwd=d, shell=True) if os.path.exists(os.path.join(d,'package-lock.json')) else subprocess.call('npm install --silent', cwd=d, shell=True)))"
 
-    # npm run build: frontend が存在する場合のみ実行
     - >
       python -c "import os,sys,subprocess;
       d='frontend';
       sys.exit(0 if not os.path.isdir(d) else subprocess.call('npm run build --silent', cwd=d, shell=True))"
 
-    # （既存の Python / React checks の下に追加）
-    - python core/openapi_contract.py
-
-    - >
-      python -c "import os,sys;
-      sys.exit(0 if not os.path.exists('contracts/openapi.json') else
-      __import__('subprocess').call([sys.executable,'core/openapi_contract.py']))"
-
-    # 失敗時ログ要約（check が失敗しても要約は残す）
-    - python core/summarize_logs.py || true
-
-
 output:
   pack: ai_context_pack.md
   audit: CHANGELOG_AI.md
   testlog: ai_test_last.log
+  report_json: ai_check_report.json

```

### $ python --version || true
```text
Python 3.13.3

```

### $ node --version || true
```text
v24.11.0

```

### $ npm --version || true
```text
11.6.1

```
## Changed files (diff-first)

### File list
```text
.github/workflows/oceansguard.yml
README.md
core/aiguard.py
templates/.aiguard.yml
```

### .github/workflows/oceansguard.yml
```text
name: OceansGuard

on:
  pull_request:
  push:
    branches: [ main ]

permissions:
  contents: read

jobs:
  guard:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout (with submodules)
        uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Upgrade pip
        run: python -m pip install --upgrade pip

      - name: Install tooling (best-effort)
        run: |
          pip install pyyaml || true
          pip install ruff mypy pytest || true
          pip install uvicorn fastapi || true

      - name: Resolve OceansGuard entry
        id: og
        run: |
          if [ -f "tools/OceansGuard/core/aiguard.py" ]; then
            echo "ENTRY=tools/OceansGuard/core/aiguard.py" >> $GITHUB_OUTPUT
          elif [ -f "core/aiguard.py" ]; then
            echo "ENTRY=core/aiguard.py" >> $GITHUB_OUTPUT
          else
            echo "OceansGuard entry not found" >&2
            exit 1
          fi

      - name: OceansGuard init (idempotent)
        run: |
          python "${{ steps.og.outputs.ENTRY }}" init --repo .

      - name: OceansGuard run (pack + check)
        run: |
          python "${{ steps.og.outputs.ENTRY }}" run --repo . --task "CI guard" --strict

```

### README.md
```text
# README.md
# OceansGuard

AI-assisted development guardrails for any repository.

## What it solves
- AI-generated changes that accidentally drop existing code
- Lack of global context (only partial files shown)
- Forgetfulness / inconsistent constraints across sessions
- No test / lint guarantees
- Secret leakage (keys/tokens) into commits
- Risky full-rewrite changes

## Core commands

### init
Create minimal guard files in target repo (idempotent; no overwrite).

python core/aiguard.py init --repo .

### pack
Generate AI context pack (diff-first).
```
python core/aiguard.py pack --repo .
```
### check
Run guard checks + configured project checks and write reports.
```
python core/aiguard.py check --repo .
```
### run
Shortcut = pack + check.
```
python core/aiguard.py run --repo . --task "your task"
```
## Strict mode
--strict makes guardrails non-negotiable:
- requires PyYAML
- fails if checks.commands is empty
- fails if contracts/openapi.json is missing/empty
```
python core/aiguard.py run --repo . --task "CI guard" --strict
```

## Submodule usage (recommended)
In your target repository:
```
git submodule add https://github.com/OceansCreative/OceansGuard.git tools/OceansGuard
python tools/OceansGuard/core/aiguard.py init --repo .
python tools/OceansGuard/core/aiguard.py run --repo . --task "初回ガード適用"
```
## Outputs
- ai_context_pack.md: single file to paste into AI chat
- ai_test_last.log: raw execution logs
- ai_check_report.json: structured result for CI/PR gating

## Git hooks (prevent committing to main)
Install with:
```
python core/install_hooks.py --repo .
```
```

### core/aiguard.py
```text
# core/aiguard.py
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Utilities
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def info(msg: str) -> None:
    print(f"[OceansGuard] {msg}")


def warn(msg: str) -> None:
    print(f"[OceansGuard][WARN] {msg}")


def die(msg: str, code: int = 1) -> None:
    raise SystemExit(f"[OceansGuard] {msg}")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def run_argv(argv, cwd=None):
    import subprocess, os

    p = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        env=os.environ.copy(),
    )

    def _decode(b: bytes) -> str:
        if not b:
            return ""
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="replace")

    p.stdout = _decode(p.stdout)
    p.stderr = _decode(p.stderr)
    return p



def run_shell(cmd, cwd=None):
    import subprocess, os

    p = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    def _decode(b: bytes) -> str:
        if not b:
            return ""
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="replace")

    p.stdout = _decode(p.stdout)
    p.stderr = _decode(p.stderr)
    return p


def safe_read_text(p: Path, max_kb: int) -> str:
    try:
        b = p.read_bytes()
    except Exception as e:
        return f"(failed to read: {e})\n"
    if len(b) > max_kb * 1024:
        return f"(skipped: too large {len(b)} bytes > {max_kb}KB)\n"
    return b.decode("utf-8", errors="replace")


def guard_root() -> Path:
    # core/aiguard.py → OceansGuard/
    return Path(__file__).resolve().parent.parent


def ensure_pyyaml(strict: bool) -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        if strict:
            die("PyYAML is required in --strict mode. Install: pip install pyyaml")
        warn("PyYAML not found. Some config-driven features may be skipped.")
        return False


# =========================
# Config (.aiguard.yml)
# =========================
@dataclass
class ContextSmall:
    include: List[str]


@dataclass
class ContextLarge:
    roots: List[str]
    exclude_dirs: List[str]
    exclude_globs: List[str]
    max_files: int
    max_kb_each: int


@dataclass
class Evidence:
    commands: List[str]


@dataclass
class Dlp:
    enable: bool
    block_on_detect: bool
    mask: bool
    allowlist_files: List[str]


@dataclass
class Guard:
    forbid_full_rewrite: bool
    allow_full_rewrite_globs: List[str]


@dataclass
class Checks:
    commands: List[str]


@dataclass
class Output:
    pack: str
    audit: str
    testlog: str
    report_json: str


@dataclass
class Config:
    version: int
    context_small: ContextSmall
    context_large: ContextLarge
    evidence: Evidence
    dlp: Dlp
    guard: Guard
    checks: Checks
    output: Output


def _dict_get(d: dict, key: str, default):
    v = d.get(key, default)
    return default if v is None else v


def load_config(repo: Path, strict: bool) -> Config:
    has_yaml = ensure_pyyaml(strict=strict)
    cfg_path = repo / ".aiguard.yml"
    if not cfg_path.exists():
        die(".aiguard.yml not found. Run init first.")

    raw: Dict[str, Any] = {}
    if has_yaml:
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            if strict:
                die(f"Failed to parse .aiguard.yml: {e}")
            warn(f"Failed to parse .aiguard.yml; using minimal defaults. ({e})")
            raw = {}
    else:
        if strict:
            die("Cannot read .aiguard.yml without PyYAML in strict mode.")

    version = int(_dict_get(raw, "version", 1))

    ctx = _dict_get(raw, "context", {})
    small = _dict_get(ctx, "small", {})
    large = _dict_get(ctx, "large", {})

    context_small = ContextSmall(include=[str(x) for x in _dict_get(small, "include", [])])
    context_large = ContextLarge(
        roots=[str(x) for x in _dict_get(large, "roots", ["backend", "app", "src", "frontend"])],
        exclude_dirs=[str(x) for x in _dict_get(large, "exclude_dirs", [
            ".git", ".venv", "venv", "node_modules", "dist", "build", ".next",
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ])],
        exclude_globs=[str(x) for x in _dict_get(large, "exclude_globs", ["**/*.min.js", "**/*.map"])],
        max_files=int(_dict_get(large, "max_files", 220)),
        max_kb_each=int(_dict_get(large, "max_kb_each", 64)),
    )

    ev = _dict_get(raw, "evidence", {})
    evidence = Evidence(commands=[str(x) for x in _dict_get(ev, "commands", [])])

    dlp_raw = _dict_get(raw, "dlp", {})
    dlp = Dlp(
        enable=bool(_dict_get(dlp_raw, "enable", True)),
        block_on_detect=bool(_dict_get(dlp_raw, "block_on_detect", True)),
        mask=bool(_dict_get(dlp_raw, "mask", True)),
        allowlist_files=[str(x) for x in _dict_get(dlp_raw, "allowlist_files", [])],
    )

    guard_raw = _dict_get(raw, "guard", {})
    guard = Guard(
        forbid_full_rewrite=bool(_dict_get(guard_raw, "forbid_full_rewrite", True)),
        allow_full_rewrite_globs=[str(x) for x in _dict_get(guard_raw, "allow_full_rewrite_globs", [])],
    )

    checks_raw = _dict_get(raw, "checks", {})
    checks = Checks(commands=[str(x) for x in _dict_get(checks_raw, "commands", [])])

    out_raw = _dict_get(raw, "output", {})
    output = Output(
        pack=str(_dict_get(out_raw, "pack", "ai_context_pack.md")),
        audit=str(_dict_get(out_raw, "audit", "CHANGELOG_AI.md")),
        testlog=str(_dict_get(out_raw, "testlog", "ai_test_last.log")),
        report_json=str(_dict_get(out_raw, "report_json", "ai_check_report.json")),
    )

    return Config(
        version=version,
        context_small=context_small,
        context_large=context_large,
        evidence=evidence,
        dlp=dlp,
        guard=guard,
        checks=checks,
        output=output,
    )


# =========================
# init
# =========================
def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        info(f"skip exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    info(f"create: {dst}")


def write_if_missing(dst: Path, content: str) -> None:
    if dst.exists():
        info(f"skip exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    info(f"create: {dst}")


def cmd_init(repo: Path) -> None:
    tpl = guard_root() / "templates" / ".aiguard.yml"
    if not tpl.exists():
        die("templates/.aiguard.yml not found in OceansGuard")
    copy_if_missing(tpl, repo / ".aiguard.yml")

    write_if_missing(
        repo / "ARCHITECTURE.md",
        "# ARCHITECTURE\n\n"
        "- レイヤ構成\n"
        "- 依存方向\n"
        "- 外部I/O（API/DB）\n\n"
        f"_generated by OceansGuard init @ {now_iso()}_\n",
    )
    write_if_missing(
        repo / "SNAPSHOT.md",
        "# SNAPSHOT\n\n"
        "- 現在の仕様\n"
        "- 既知の制約\n"
        "- 触ってはいけない領域\n\n"
        f"_generated by OceansGuard init @ {now_iso()}_\n",
    )

    contracts = repo / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    write_if_missing(
        contracts / "README.md",
        "# Contracts\n\n"
        "このディレクトリには、守るべき契約（スキーマ/仕様）を置きます。\n\n"
        "- OpenAPI: contracts/openapi.json（または openapi.yaml）\n"
        "- DB schema snapshot\n"
        "- DTO/型\n",
    )
    if not (contracts / "openapi.json").exists():
        (contracts / "openapi.json").write_text("", encoding="utf-8")
        info("create: contracts/openapi.json (empty)")

    info("init completed")


# =========================
# pack (diff-first)
# =========================
def pack_git_section(repo: Path) -> str:
    buf: List[str] = []
    buf.append("## Git\n")

    st = run_argv(["git", "status", "--porcelain=v1"], cwd=repo)
    buf.append("\n### git status --porcelain=v1\n```text\n")
    buf.append(st.stdout or "")
    buf.append("\n```\n")

    df = run_argv(["git", "diff"], cwd=repo)
    diff_text = df.stdout or ""
    buf.append("\n### git diff\n```diff\n")
    if len(diff_text) > 200_000:
        buf.append(diff_text[:200_000])
        buf.append("\n... (truncated)\n")
    else:
        buf.append(diff_text)
    buf.append("\n```\n")

    return "".join(buf)


def glob_files(root: Path, pattern: str) -> List[Path]:
    return [p for p in root.glob(pattern) if p.is_file()]


def should_exclude(path: Path, repo: Path, exclude_dirs: List[str], exclude_globs: List[str]) -> bool:
    rel = path.relative_to(repo).as_posix()
    parts = set(Path(rel).parts)
    if any(d in parts for d in exclude_dirs):
        return True
    for g in exclude_globs:
        if fnmatch.fnmatch(rel, g):
            return True
    return False


def collect_small(repo: Path, cfg: Config) -> List[Path]:
    out: List[Path] = []
    for pat in cfg.context_small.include:
        out.extend(glob_files(repo, pat))
    seen = set()
    uniq: List[Path] = []
    for p in sorted(out, key=lambda x: x.relative_to(repo).as_posix()):
        rel = p.relative_to(repo).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append(p)
    return uniq


def collect_changed_files(repo: Path) -> List[Path]:
    cp = run_argv(["git", "diff", "--name-only"], cwd=repo)
    files = []
    for ln in (cp.stdout or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = repo / ln
        if p.exists() and p.is_file():
            files.append(p)
    return files


def collect_large(repo: Path, cfg: Config) -> List[Path]:
    files: List[Path] = []
    roots = cfg.context_large.roots[:] if cfg.context_large.roots else ["."]
    for r in roots:
        base = (repo / r).resolve()
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs):
                continue
            files.append(p)
    files = sorted(files, key=lambda x: x.relative_to(repo).as_posix())
    if len(files) > cfg.context_large.max_files:
        files = files[: cfg.context_large.max_files]
    return files


def evidence_section(repo: Path, cfg: Config) -> str:
    buf: List[str] = []
    buf.append("## Evidence\n")
    for cmd in cfg.evidence.commands:
        buf.append(f"\n### $ {cmd}\n")
        cp = run_shell(cmd, cwd=repo)
        buf.append("```text\n")
        buf.append(cp.stdout or "")
        buf.append("\n```\n")
    return "".join(buf)


def pack_files(repo: Path, files: List[Path], max_kb_each: int, title: str) -> str:
    buf: List[str] = []
    buf.append(f"## {title}\n")
    buf.append("\n### File list\n```text\n")
    for p in files:
        buf.append(p.relative_to(repo).as_posix() + "\n")
    buf.append("```\n\n")

    for p in files:
        rel = p.relative_to(repo).as_posix()
        buf.append(f"### {rel}\n")
        buf.append("```text\n")
        buf.append(safe_read_text(p, max_kb=max_kb_each))
        buf.append("\n```\n\n")
    return "".join(buf)


def cmd_pack(repo: Path, strict: bool) -> None:
    cfg = load_config(repo, strict=strict)
    out_path = repo / cfg.output.pack

    small_files = collect_small(repo, cfg)

    # diff-first: 変更ファイルを優先的に添付（除外規則も適用）
    changed = [
        p for p in collect_changed_files(repo)
        if not should_exclude(p, repo, cfg.context_large.exclude_dirs, cfg.context_large.exclude_globs)
    ]
    # large は補助（上限付き）
    large_files = collect_large(repo, cfg)

    buf: List[str] = []
    buf.append("# OceansGuard Context Pack\n\n")
    buf.append(f"- generated_at: {now_iso()}\n")
    buf.append(f"- repo: {repo}\n")
    buf.append(f"- config_version: {cfg.version}\n")
    buf.append(f"- mode: {'strict' if strict else 'normal'}\n\n")

    buf.append(pack_git_section(repo))
    buf.append(evidence_section(repo, cfg))

    if changed:
        buf.append(pack_files(repo, changed, max_kb_each=cfg.context_large.max_kb_each, title="Changed files (diff-first)"))
    if small_files:
        buf.append(pack_files(repo, small_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (small)"))
    if large_files:
        buf.append(pack_files(repo, large_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (large, capped)"))

    out_path.write_text("".join(buf), encoding="utf-8")
    info(f"pack written: {out_path}")


# =========================
# Guard: full rewrite detection
# =========================
def detect_full_rewrite(repo: Path, forbid: bool, allow_globs: List[str]) -> Optional[str]:
    if not forbid:
        return None

    cp = run_argv(["git", "diff", "--numstat"], cwd=repo)
    suspicious: List[str] = []

    def _allowed(path_str: str) -> bool:
        # numstat の file は基本 posix だが、念のため正規化
        s = path_str.replace("\\", "/")
        for g in allow_globs or []:
            if fnmatch.fnmatch(s, g):
                return True
        return False

    for ln in (cp.stdout or "").splitlines():
        parts = ln.split("\t")
        if len(parts) != 3:
            continue
        add_s, del_s, file_s = parts
        if add_s == "-" or del_s == "-":
            continue

        if _allowed(file_s):
            continue

        try:
            add_n = int(add_s)
            del_n = int(del_s)
        except ValueError:
            continue

        if (add_n + del_n) >= 800 and add_n >= 300 and del_n >= 300:
            suspicious.append(f"{file_s} (add={add_n}, del={del_n})")

    if suspicious:
        return "Possible full rewrite detected:\n" + "\n".join(f"- {s}" for s in suspicious)
    return None



# =========================
# DLP
# =========================
SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("JWT_LIKE", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("PASSWORD_ASSIGN", re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]")),
]


def is_allowlisted(rel_posix: str, allowlist: List[str]) -> bool:
    for pat in allowlist:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def scan_dlp(repo: Path, enable: bool, allowlist: List[str], mask: bool, strict: bool) -> List[str]:
    if not enable:
        return []
    max_bytes = 256 * 1024  # 256KB

    cp = run_argv(["git", "ls-files"], cwd=repo)
    hits: List[str] = []

    for rel in (cp.stdout or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if is_allowlisted(rel, allowlist):
            continue
        p = repo / rel
        if not p.exists() or not p.is_file():
            continue
        try:
            b = p.read_bytes()
        except Exception:
            continue
        if len(b) > max_bytes:
            continue
        s = b.decode("utf-8", errors="ignore")
        for name, pat in SECRET_PATTERNS:
            m = pat.search(s)
            if not m:
                continue
            sample = m.group(0)
            if mask:
                sample = f"<masked:{sha256_text(sample)[:12]}>"
            hits.append(f"{name}: {rel}: {sample}")

    # strict は「検知ルールのスキップ」ではなく、検知をより厳密にする余地（今は同等）
    return hits


# =========================
# OpenAPI contract
# =========================
def openapi_contract_check(repo: Path, strict: bool) -> Optional[str]:
    p = repo / "contracts" / "openapi.json"
    if not p.exists():
        if strict:
            return "contracts/openapi.json is required in strict mode (can be empty only in normal mode)"
        return None

    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        if strict:
            return "contracts/openapi.json is empty in strict mode"
        return None

    try:
        obj = json.loads(txt)
    except Exception as e:
        return f"contracts/openapi.json is not valid JSON: {e}"

    if not isinstance(obj, dict):
        return "contracts/openapi.json must be a JSON object"
    if "openapi" not in obj and "swagger" not in obj:
        return "contracts/openapi.json missing 'openapi' (or 'swagger') field"
    return None


# =========================
# check (report json)
# =========================
def cmd_check(repo: Path, strict: bool) -> None:
    cfg = load_config(repo, strict=strict)

    report: Dict[str, Any] = {
        "tool": "OceansGuard",
        "generated_at": now_iso(),
        "repo": str(repo),
        "mode": "strict" if strict else "normal",
        "checks": [],
        "dlp_hits": [],
        "guard": {"full_rewrite": None},
        "openapi_contract": None,
        "status": "pass",
    }

    # 1) full rewrite guard
    fr = detect_full_rewrite(repo, forbid=cfg.guard.forbid_full_rewrite, allow_globs=cfg.guard.allow_full_rewrite_globs)

    if fr:
        report["guard"]["full_rewrite"] = fr
        report["status"] = "fail"
        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        die(fr)

    # 2) DLP
    dlp_hits = scan_dlp(
        repo,
        enable=cfg.dlp.enable,
        allowlist=cfg.dlp.allowlist_files,
        mask=cfg.dlp.mask,
        strict=strict,
    )
    report["dlp_hits"] = dlp_hits
    if dlp_hits and cfg.dlp.block_on_detect:
        report["status"] = "fail"
        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        die("DLP detected potential secrets:\n" + "\n".join(f"- {h}" for h in dlp_hits))

    # 3) OpenAPI contract
    oc = openapi_contract_check(repo, strict=strict)
    report["openapi_contract"] = oc
    if oc:
        report["status"] = "fail"
        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        die(oc)

    # 4) checks.commands
    log_path = repo / cfg.output.testlog
    logs: List[str] = []
    logs.append(f"[OceansGuard] check started @ {now_iso()}\n")

    failed: List[Tuple[str, int]] = []

    for i, cmd in enumerate(cfg.checks.commands, start=1):
        cmd = (cmd or "").strip()
        if not cmd:
            continue
        info(f"check[{i}] {cmd}")
        cp = run_shell(cmd, cwd=repo)
        entry = {
            "index": i,
            "command": cmd,
            "exit_code": cp.returncode,
        }
        report["checks"].append(entry)

        logs.append(f"\n=== check[{i}] {cmd} ===\n")
        logs.append(cp.stdout or "")
        logs.append(f"\n[exit_code] {cp.returncode}\n")

        if cp.returncode != 0:
            failed.append((cmd, cp.returncode))

    log_path.write_text("".join(logs), encoding="utf-8")
    info(f"testlog written: {log_path}")

    # strict: checks.commands が空なら fail（「テスト保証」を強制）
    if strict and not cfg.checks.commands:
        report["status"] = "fail"
        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        die("checks.commands is empty in strict mode")

    if failed:
        report["status"] = "fail"
        (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = "\n".join([f"- ({rc}) {cmd}" for cmd, rc in failed])
        die("check failed:\n" + summary)

    (repo / cfg.output.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"report written: {repo / cfg.output.report_json}")
    info("check passed")


# =========================
# run
# =========================
def cmd_run(repo: Path, task: str, strict: bool) -> None:
    if task.strip():
        info(f"run task: {task}")
    cmd_pack(repo, strict=strict)
    cmd_check(repo, strict=strict)


def main() -> None:
    ap = argparse.ArgumentParser(prog="aiguard")
    ap.add_argument("command", choices=["init", "pack", "check", "run"])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--task", default="")
    ap.add_argument("--strict", action="store_true", help="Do not allow skipping; fail if prerequisites missing.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    if args.command == "init":
        cmd_init(repo)
        return
    if args.command == "pack":
        cmd_pack(repo, strict=args.strict)
        return
    if args.command == "check":
        cmd_check(repo, strict=args.strict)
        return
    if args.command == "run":
        cmd_run(repo, task=args.task, strict=args.strict)
        return

    die("unknown command")


if __name__ == "__main__":
    main()

```

### templates/.aiguard.yml
```text
version: 1

context:
  small:
    include:
      - ARCHITECTURE.md
      - SNAPSHOT.md
      - README.md
      - pyproject.toml
      - package.json
      - frontend/package.json
      - contracts/**
  large:
    roots: [backend, app, src, frontend]
    exclude_dirs:
      - .git
      - .venv
      - venv
      - node_modules
      - dist
      - build
      - .next
      - __pycache__
      - __pypackages__
      - .pytest_cache
      - .mypy_cache
      - .ruff_cache
    exclude_globs: ["**/*.min.js", "**/*.map"]
    max_files: 220
    max_kb_each: 64

evidence:
  commands:
    - git status --porcelain=v1 || true
    - git diff || true
    - python --version || true
    - node --version || true
    - npm --version || true

dlp:
  enable: true
  block_on_detect: true
  mask: true
  allowlist_files:
    - ".env.example"
    - "frontend/.env.example"

guard:
  forbid_full_rewrite: true
  allow_full_rewrite_globs:
    - "core/aiguard.py"
    - ".github/workflows/oceansguard.yml"

checks:
  commands:
    # =========================
    # Backend (Python)
    # =========================
    - >
      python -c "import os,sys,subprocess;
      targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
      sys.exit(subprocess.call([sys.executable,'-m','compileall',*targets]) if targets else 0)"

    - >
      python -c "import os,sys,subprocess,importlib.util;
      targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
      has=importlib.util.find_spec('ruff') is not None;
      sys.exit(subprocess.call([sys.executable,'-m','ruff','check',*targets]) if (has and targets) else 0)"

    - >
      python -c "import os,sys,subprocess,importlib.util;
      targets=[d for d in ('backend','app','src') if os.path.isdir(d)];
      has=importlib.util.find_spec('mypy') is not None;
      sys.exit(subprocess.call([sys.executable,'-m','mypy',*targets]) if (has and targets) else 0)"

    - >
      python -c "import os,sys,subprocess,importlib.util;
      has=importlib.util.find_spec('pytest') is not None;
      has_tests=any(os.path.isdir(p) for p in ('tests','backend/tests','app/tests','src/tests'));
      sys.exit(subprocess.call([sys.executable,'-m','pytest','-q']) if (has and has_tests) else 0)"

    # =========================
    # Frontend (React)
    # =========================
    - >
      python -c "import os,sys,subprocess;
      d='frontend';
      sys.exit(0 if not os.path.isdir(d) else (subprocess.call('npm ci --silent', cwd=d, shell=True) if os.path.exists(os.path.join(d,'package-lock.json')) else subprocess.call('npm install --silent', cwd=d, shell=True)))"

    - >
      python -c "import os,sys,subprocess;
      d='frontend';
      sys.exit(0 if not os.path.isdir(d) else subprocess.call('npm run build --silent', cwd=d, shell=True))"

output:
  pack: ai_context_pack.md
  audit: CHANGELOG_AI.md
  testlog: ai_test_last.log
  report_json: ai_check_report.json

```

## Context (small)

### File list
```text
ARCHITECTURE.md
README.md
SNAPSHOT.md
contracts/README.md
contracts/openapi.json
```

### ARCHITECTURE.md
```text
# ARCHITECTURE

- レイヤ構成
- 依存方向
- 外部I/O（API/DB）

_generated by OceansGuard init @ 2025-12-31T10:35:26_

```

### README.md
```text
# README.md
# OceansGuard

AI-assisted development guardrails for any repository.

## What it solves
- AI-generated changes that accidentally drop existing code
- Lack of global context (only partial files shown)
- Forgetfulness / inconsistent constraints across sessions
- No test / lint guarantees
- Secret leakage (keys/tokens) into commits
- Risky full-rewrite changes

## Core commands

### init
Create minimal guard files in target repo (idempotent; no overwrite).

python core/aiguard.py init --repo .

### pack
Generate AI context pack (diff-first).
```
python core/aiguard.py pack --repo .
```
### check
Run guard checks + configured project checks and write reports.
```
python core/aiguard.py check --repo .
```
### run
Shortcut = pack + check.
```
python core/aiguard.py run --repo . --task "your task"
```
## Strict mode
--strict makes guardrails non-negotiable:
- requires PyYAML
- fails if checks.commands is empty
- fails if contracts/openapi.json is missing/empty
```
python core/aiguard.py run --repo . --task "CI guard" --strict
```

## Submodule usage (recommended)
In your target repository:
```
git submodule add https://github.com/OceansCreative/OceansGuard.git tools/OceansGuard
python tools/OceansGuard/core/aiguard.py init --repo .
python tools/OceansGuard/core/aiguard.py run --repo . --task "初回ガード適用"
```
## Outputs
- ai_context_pack.md: single file to paste into AI chat
- ai_test_last.log: raw execution logs
- ai_check_report.json: structured result for CI/PR gating

## Git hooks (prevent committing to main)
Install with:
```
python core/install_hooks.py --repo .
```
```

### SNAPSHOT.md
```text
# SNAPSHOT

- 現在の仕様
- 既知の制約
- 触ってはいけない領域

_generated by OceansGuard init @ 2025-12-31T10:35:26_

```

### contracts/README.md
```text
# Contracts

このディレクトリには、守るべき契約（スキーマ/仕様）を置きます。

- OpenAPI: contracts/openapi.json（または openapi.yaml）
- DB schema snapshot
- DTO/型

```

### contracts/openapi.json
```text
{
  "openapi": "3.0.3",
  "info": {
    "title": "OceansGuard Placeholder API",
    "version": "0.0.0"
  },
  "paths": {}
}

```

