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
