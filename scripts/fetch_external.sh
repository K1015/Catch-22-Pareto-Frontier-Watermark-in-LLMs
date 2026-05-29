#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clone the upstream watermark method repositories into $CATCH22_EXTERNAL_DIR
# at pinned commits. These provide the genuine SemStamp / PMark /
# HeavyWater / SimplexWater implementations that the pipeline adapts.
#
# Usage:
#   source env.sh
#   scripts/fetch_external.sh
#
# The external/ directory is git-ignored; re-running is idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail

# Load configuration if the caller did not already source it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${CATCH22_EXTERNAL_DIR:-}" ]]; then
  if [[ -f "$REPO_ROOT/env.sh" ]]; then source "$REPO_ROOT/env.sh"; else source "$REPO_ROOT/env.example.sh"; fi
fi

EXTERNAL_DIR="${CATCH22_EXTERNAL_DIR:-$REPO_ROOT/external}"
mkdir -p "$EXTERNAL_DIR"

# name|url|pinned-commit
REPOS=(
  "SemStamp|https://github.com/abehou/SemStamp|97db73d11fd80f376a02b0a604d500627622f7e6"
  "PMark|https://github.com/PMark-repo/PMark|75140fe7142f88c51ba50984d9eb07f44f5961b3"
  "HeavyWater_SimplexWater|https://github.com/DorTsur/HeavyWater_SimplexWater|5be1467810f607e0f844fe7ce57a6214c92d6160"
)

clone_pinned() {
  local name="$1" url="$2" commit="$3"
  local dest="$EXTERNAL_DIR/$name"
  if [[ -d "$dest/.git" ]]; then
    echo "[fetch_external] $name already present -> checking out $commit"
    git -C "$dest" fetch --quiet origin || true
  else
    echo "[fetch_external] cloning $name from $url"
    git clone --quiet "$url" "$dest"
  fi
  git -C "$dest" checkout --quiet "$commit"
  # SemStamp ships its evaluation data as a submodule.
  if [[ -f "$dest/.gitmodules" ]]; then
    git -C "$dest" submodule update --init --recursive --quiet || \
      echo "[fetch_external] WARN: submodule init failed for $name (data-only, usually optional)"
  fi
  echo "[fetch_external] $name @ $(git -C "$dest" rev-parse --short HEAD)"
}

for spec in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<<"$spec"
  clone_pinned "$name" "$url" "$commit"
done

echo "[fetch_external] Done. Upstream repos are in $EXTERNAL_DIR"
echo "[fetch_external] Install their Python requirements into the same environment, e.g.:"
echo "                 pip install -r $EXTERNAL_DIR/SemStamp/requirements.txt"
