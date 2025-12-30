# core/aiguard.py
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# =========================
# 基本ユーティリティ
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def info(msg: str) -> None:
    print(f"[OceansGuard] {msg}")


def warn(msg: str) -> None:
    print(f"[OceansGuard][WARN] {msg}")


def die(msg: str, code: int = 1) -> None:
    raise SystemExit(f"[OceansGuard] {msg}")


def run(cmd: List[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        env=os.environ.copy(),
    )


def run_shell(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )


def safe_read_text(p: Path, max_kb: int) -> str:
    try:
        b = p.read_bytes()
    except Exception as e:
        return f"(failed to read: {e})\n"
    if len(b) > max_kb * 1024:
        return f"(skipped: too large {len(b)} bytes > {max_kb}KB)\n"
    try:
        return b.decode("utf-8", errors="replace")
    except Exception as e:
        return f"(failed to decode: {e})\n"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def guard_root() -> Path:
    # core/aiguard.py → OceansGuard/
    return Path(__file__).resolve().parent.parent


def ensure_pyyaml() -> None:
    """
    ユーザー負担ゼロを優先して、PyYAML が無ければ best-effort で導入する。
    CI でもローカルでも動く方針。
    """
    try:
        import yaml  # noqa: F401
        return
    except Exception:
        pass

    warn("PyYAML not found. Trying to install pyyaml (best-effort).")
    try:
        # python -m pip install pyyaml
        cp = run([os.environ.get("PYTHON", "python"), "-m", "pip", "install", "pyyaml"], cwd=Path.cwd())
        if cp.returncode != 0:
            warn(cp.stdout)
            warn("Failed to install pyyaml. Some features may be skipped.")
            return
    except Exception as e:
        warn(f"pip install pyyaml failed: {e}")
        return


# =========================
# 設定（.aiguard.yml）
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
    max_files: int
    max_lines: int
    forbid_full_rewrite: bool


@dataclass
class Checks:
    commands: List[str]


@dataclass
class Output:
    pack: str
    audit: str
    testlog: str


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


def load_config(repo: Path) -> Config:
    ensure_pyyaml()
    cfg_path = repo / ".aiguard.yml"
    if not cfg_path.exists():
        die(".aiguard.yml not found. Run `python <OceansGuard>/core/aiguard.py init --repo <repo>` first.")

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        die(f"Failed to parse .aiguard.yml: {e}")

    version = int(_dict_get(raw, "version", 1))

    ctx = _dict_get(raw, "context", {})
    small = _dict_get(ctx, "small", {})
    large = _dict_get(ctx, "large", {})

    context_small = ContextSmall(include=[str(x) for x in _dict_get(small, "include", [])])
    context_large = ContextLarge(
        roots=[str(x) for x in _dict_get(large, "roots", [])],
        exclude_dirs=[str(x) for x in _dict_get(large, "exclude_dirs", [])],
        exclude_globs=[str(x) for x in _dict_get(large, "exclude_globs", [])],
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
        max_files=int(_dict_get(guard_raw, "max_files", 10)),
        max_lines=int(_dict_get(guard_raw, "max_lines", 400)),
        forbid_full_rewrite=bool(_dict_get(guard_raw, "forbid_full_rewrite", True)),
    )

    checks_raw = _dict_get(raw, "checks", {})
    checks = Checks(commands=[str(x) for x in _dict_get(checks_raw, "commands", [])])

    out_raw = _dict_get(raw, "output", {})
    output = Output(
        pack=str(_dict_get(out_raw, "pack", "ai_context_pack.md")),
        audit=str(_dict_get(out_raw, "audit", "CHANGELOG_AI.md")),
        testlog=str(_dict_get(out_raw, "testlog", "ai_test_last.log")),
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
# init（コピー/雛形生成）
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
    # openapi.json は「空でもOK」運用にする（未導入PJでも害がない）
    if not (contracts / "openapi.json").exists():
        (contracts / "openapi.json").write_text("", encoding="utf-8")
        info("create: contracts/openapi.json (empty)")

    info("init completed")


# =========================
# pack（AIに渡すコンテキスト束ね）
# =========================
def glob_files(root: Path, pattern: str) -> List[Path]:
    # ** を含む想定
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

    # 安定順（相対パス）
    files = sorted(files, key=lambda x: x.relative_to(repo).as_posix())

    # 上限
    if len(files) > cfg.context_large.max_files:
        files = files[: cfg.context_large.max_files]
    return files


def collect_small(repo: Path, cfg: Config) -> List[Path]:
    out: List[Path] = []
    for pat in cfg.context_small.include:
        for p in glob_files(repo, pat):
            out.append(p)
    # unique & stable
    seen = set()
    uniq: List[Path] = []
    for p in sorted(out, key=lambda x: x.relative_to(repo).as_posix()):
        rel = p.relative_to(repo).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append(p)
    return uniq


def evidence_section(repo: Path, cfg: Config) -> str:
    buf = []
    buf.append("## Evidence\n")
    for cmd in cfg.evidence.commands:
        buf.append(f"\n### $ {cmd}\n")
        cp = run_shell(cmd, cwd=repo)
        buf.append("```text\n")
        buf.append(cp.stdout or "")
        buf.append("\n```\n")
    return "".join(buf)


def git_diff_section(repo: Path) -> str:
    # CI/ローカルどちらも動く安全版
    buf = []
    buf.append("## Git\n")
    st = run(["git", "status", "--porcelain=v1"], cwd=repo)
    buf.append("\n### git status --porcelain=v1\n```text\n")
    buf.append(st.stdout or "")
    buf.append("\n```\n")

    df = run(["git", "diff"], cwd=repo)
    buf.append("\n### git diff\n```diff\n")
    # diff が巨大化するとAIが壊れるので適度に切る
    diff_text = df.stdout or ""
    if len(diff_text) > 200_000:
        buf.append(diff_text[:200_000])
        buf.append("\n... (truncated)\n")
    else:
        buf.append(diff_text)
    buf.append("\n```\n")
    return "".join(buf)


def pack_files_section(repo: Path, files: List[Path], max_kb_each: int, title: str) -> str:
    buf = []
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


def cmd_pack(repo: Path) -> None:
    cfg = load_config(repo)
    out_path = repo / cfg.output.pack

    small_files = collect_small(repo, cfg)
    large_files = collect_large(repo, cfg)

    buf = []
    buf.append("# OceansGuard Context Pack\n\n")
    buf.append(f"- generated_at: {now_iso()}\n")
    buf.append(f"- repo: {repo}\n")
    buf.append(f"- config_version: {cfg.version}\n")
    buf.append(f"- note: This pack is intended for AI-assisted changes with guardrails.\n\n")

    buf.append(git_diff_section(repo))
    buf.append(evidence_section(repo, cfg))

    if small_files:
        buf.append(pack_files_section(repo, small_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (small)"))
    if large_files:
        buf.append(pack_files_section(repo, large_files, max_kb_each=cfg.context_large.max_kb_each, title="Context (large, capped)"))

    out_path.write_text("".join(buf), encoding="utf-8")
    info(f"pack written: {out_path}")


# =========================
# Guard: 禁止系（全置換/危険diff）
# =========================
def detect_full_rewrite(repo: Path, cfg: Config) -> Optional[str]:
    if not cfg.guard.forbid_full_rewrite:
        return None

    # numstat: add del file
    cp = run(["git", "diff", "--numstat"], cwd=repo)
    lines = (cp.stdout or "").splitlines()
    suspicious: List[str] = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) != 3:
            continue
        add_s, del_s, file_s = parts
        if add_s == "-" or del_s == "-":
            continue
        try:
            add_n = int(add_s)
            del_n = int(del_s)
        except ValueError:
            continue

        # 目安：大きいファイルで add+del が巨大、かつ双方が大きい → 全置換の疑い
        if (add_n + del_n) >= 800 and add_n >= 300 and del_n >= 300:
            suspicious.append(f"{file_s} (add={add_n}, del={del_n})")

    if suspicious:
        return "Possible full rewrite detected:\n" + "\n".join(f"- {s}" for s in suspicious)
    return None


# =========================
# DLP（秘密情報・鍵・トークンの検知）
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


def mask_hit(text: str) -> str:
    # 露出防止：sha256のみ残す
    return f"<masked:{sha256_text(text)[:12]}>"


def scan_dlp(repo: Path, cfg: Config) -> List[str]:
    if not cfg.dlp.enable:
        return []

    # 変更されたファイル中心にしたいが、CIでも安全に動くよう「追跡対象のうちテキストだけ」軽く走査
    # サイズ上限で暴走防止
    max_bytes = 256 * 1024  # 256KB
    hits: List[str] = []

    cp = run(["git", "ls-files"], cwd=repo)
    files = (cp.stdout or "").splitlines()
    for rel in files:
        rel_posix = rel.strip()
        if not rel_posix:
            continue
        if is_allowlisted(rel_posix, cfg.dlp.allowlist_files):
            continue
        p = repo / rel_posix
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
            if cfg.dlp.mask:
                sample = mask_hit(sample)
            hits.append(f"{name}: {rel_posix}: {sample}")
    return hits


# =========================
# OpenAPI 契約チェック（空ならスキップ）
# =========================
def openapi_contract_check(repo: Path) -> Optional[str]:
    p = repo / "contracts" / "openapi.json"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        # 空は「未導入」扱いでスキップ（あなたの要件）
        return None

    # 形式妥当性だけ最小チェック（JSON）
    import json

    try:
        obj = json.loads(txt)
    except Exception as e:
        return f"contracts/openapi.json is not valid JSON: {e}"

    # openapi/version の最低限
    if not isinstance(obj, dict):
        return "contracts/openapi.json must be a JSON object"
    if "openapi" not in obj and "swagger" not in obj:
        return "contracts/openapi.json missing 'openapi' (or 'swagger') field"
    return None


# =========================
# check（コマンド実行 + ガード + DLP + 契約）
# =========================
def cmd_check(repo: Path) -> None:
    cfg = load_config(repo)

    # 1) full rewrite ガード
    fr = detect_full_rewrite(repo, cfg)
    if fr:
        die(fr)

    # 2) DLP
    dlp_hits = scan_dlp(repo, cfg)
    if dlp_hits and cfg.dlp.block_on_detect:
        msg = "DLP detected potential secrets:\n" + "\n".join(f"- {h}" for h in dlp_hits)
        die(msg)

    # 3) OpenAPI contract
    oc = openapi_contract_check(repo)
    if oc:
        die(oc)

    # 4) checks.commands を順に実行
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
        logs.append(f"\n=== check[{i}] {cmd} ===\n")
        logs.append(cp.stdout or "")
        logs.append(f"\n[exit_code] {cp.returncode}\n")
        if cp.returncode != 0:
            failed.append((cmd, cp.returncode))

    # 常にログを残す（あなたのストレス軽減）
    log_path.write_text("".join(logs), encoding="utf-8")
    info(f"testlog written: {log_path}")

    if failed:
        summary = "\n".join([f"- ({rc}) {cmd}" for cmd, rc in failed])
        die("check failed:\n" + summary)

    info("check passed")


# =========================
# run（AI作業の“型”を固定：pack→(作業)→check）
# =========================
def cmd_run(repo: Path, task: str) -> None:
    """
    run は「あなたの作業ストレスを下げる」ための導線。
    - まず pack を作ってAIに渡す材料を確定
    - task はログに残す（作業目的の固定）
    - 最後に check で締める
    """
    if not task.strip():
        warn("run: --task is empty (still runs pack/check)")
    else:
        info(f"run task: {task}")

    cmd_pack(repo)
    cmd_check(repo)


def main() -> None:
    ap = argparse.ArgumentParser(prog="aiguard")
    ap.add_argument("command", choices=["init", "pack", "check", "run"])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--task", default="")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    if args.command == "init":
        cmd_init(repo)
        return
    if args.command == "pack":
        cmd_pack(repo)
        return
    if args.command == "check":
        cmd_check(repo)
        return
    if args.command == "run":
        cmd_run(repo, task=args.task)
        return

    die("unknown command")


if __name__ == "__main__":
    main()
