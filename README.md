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