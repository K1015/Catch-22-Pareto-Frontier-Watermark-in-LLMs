#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Mirror the local repository to a remote cluster with rsync.
#
# Set CATCH22_REMOTE in env.sh to your target, e.g.
#   export CATCH22_REMOTE="user@host:/absolute/path/on/cluster/"
# then run:
#   source env.sh
#   scripts/sync_to_cluster.sh
#
# Run artifacts, caches, and local config are excluded so only code/manifests
# are pushed.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${CATCH22_REMOTE:-}" ]]; then
  if [[ -f "$REPO_ROOT/env.sh" ]]; then source "$REPO_ROOT/env.sh"; else source "$REPO_ROOT/env.example.sh"; fi
fi

LOCAL_ROOT="${CATCH22_ROOT:-$REPO_ROOT}"
REMOTE_ROOT="${CATCH22_REMOTE:?Set CATCH22_REMOTE in env.sh, e.g. user@host:/path/on/cluster/}"

rsync -av \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'external/' \
  --exclude 'hf_cache/' \
  --exclude 'results/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude 'env.sh' \
  "$LOCAL_ROOT/" \
  "$REMOTE_ROOT"
