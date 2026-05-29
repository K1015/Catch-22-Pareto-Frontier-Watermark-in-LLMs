#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup for the Catch-22 reproduction pipeline.
#
#   1. installs the pipeline's Python dependencies (requirements.txt)
#   2. downloads the NLTK corpora used by the attack suite
#   3. clones the upstream method repos (scripts/fetch_external.sh)
#
# Assumes you have already created and activated the Python environment named
# in CATCH22_CONDA_ENV (conda) or an equivalent venv. Run:
#
#   source env.sh
#   scripts/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${CATCH22_ROOT:-}" ]]; then
  if [[ -f "$REPO_ROOT/env.sh" ]]; then source "$REPO_ROOT/env.sh"; else source "$REPO_ROOT/env.example.sh"; fi
fi

PY="${CATCH22_PYTHON:-python3}"

echo "[setup] installing pipeline requirements"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[setup] downloading NLTK corpora (punkt, wordnet, omw-1.4)"
"$PY" - <<'PYEOF'
import nltk
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as exc:  # pragma: no cover
        print(f"[setup] WARN: could not download {pkg}: {exc}")
PYEOF

echo "[setup] cloning upstream method repositories"
bash "$SCRIPT_DIR/fetch_external.sh"

echo "[setup] Done. Next: validate a config, e.g."
echo "        $PY common/validate_experiment.py --manifest experiments/mistral7b_appendix/manifest.json"
