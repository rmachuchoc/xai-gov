#!/usr/bin/env bash
# Move the pre-remaster modules out of the package.
#
# Every one of them is rewritten in stages 2-5 (see appendix B of the
# protocol). Keeping both generations under src/xai_gov/ would let a stale
# module shadow its replacement on import, which is the kind of bug that
# costs a day to find. They are preserved verbatim under legacy/ so the
# pilot results stay reproducible from the old code if ever needed.
#
# Run once, from the project root:
#     bash scripts/retire_legacy.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEGACY_DIRS=(agents simulation policies xai conformal governance kpis orchestration)
LEGACY_FILES=(core/seeds.py)   # superseded by the SeedBundle contract

mkdir -p legacy/xai_gov

moved=0
for dir in "${LEGACY_DIRS[@]}"; do
  src="src/xai_gov/$dir"
  [ -d "$src" ] || continue
  # A stage may already have delivered the replacement; never overwrite it.
  if [ -e "legacy/xai_gov/$dir" ]; then
    echo "skip  legacy/xai_gov/$dir already exists"
    continue
  fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git mv "$src" "legacy/xai_gov/$dir" 2>/dev/null || mv "$src" "legacy/xai_gov/$dir"
  else
    mv "$src" "legacy/xai_gov/$dir"
  fi
  echo "moved $src -> legacy/xai_gov/$dir"
  moved=$((moved + 1))
done

for file in "${LEGACY_FILES[@]}"; do
  src="src/xai_gov/$file"
  [ -f "$src" ] || continue
  head -1 "$src" | grep -q "Deterministic seeding" && continue   # already the new one
done

# Stray caches from the retired modules would still be importable.
find src -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

cat > legacy/README.md <<'EOF'
# Pre-remaster code

The modules here are the first-delivery implementation, kept verbatim so
the nineteen pilot runs in `outputs/runs/` remain reproducible from the
code that produced them.

They are **not** part of the package: nothing under `src/` imports them,
`ruff` and `mypy` exclude them, and `pytest` does not collect them. Each
is superseded as its stage lands:

| Legacy module | Replaced in | By |
| --- | --- | --- |
| `simulation/engine.py` | stage 2 | multi-echelon Dec-POMDP twin |
| `agents/`, `policies/` | stage 2 | six budget-matched policy families |
| `kpis/` | stage 2-3 | six indicator layers incl. guarantees |
| `conformal/` | stage 3 | adaptive conformal inference + martingales |
| `governance/` | stage 3 | constrained POMDP over beliefs |
| `orchestration/` | stage 3 | experiment runner and campaigns |
| `xai/` | stage 4 | structural causal model and causal recourse |

Do not fix bugs here. Fix them in the replacement.
EOF

echo
echo "retired $moved module group(s); see legacy/README.md"
echo "next: make check"
