# Makefile（project root）
# OceansGuard を submodule で tools/OceansGuard に配置している前提

OCEANSGUARD := python tools/OceansGuard/core/aiguard.py

.PHONY: ai:init ai:pack ai:check ai:run

ai:init:
	$(OCEANSGUARD) init

ai:pack:
	$(OCEANSGUARD) pack

ai:check:
	$(OCEANSGUARD) check

ai:run:
	$(OCEANSGUARD) run --task "$(TASK)"
