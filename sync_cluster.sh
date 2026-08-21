#!/usr/bin/env bash
#
# Sync this repo with the RWTH cluster.
#
#   ./sync_cluster.sh push        show what an upload would change (dry run)
#   ./sync_cluster.sh push --go   actually upload
#   ./sync_cluster.sh pull-logs   download onereplay/slurm/outputs
#
# push mirrors with --delete so a file moved locally does not survive on the
# cluster as a stale duplicate, and reads its exclusions from .rsync-exclude.
#
# The most important exclusion is onereplay/slurm/outputs/. Slurm keeps a
# running job's stdout open on those files, and rsync installs a file by
# renaming a temp copy over it. The rename swaps the inode, the job's file
# descriptor goes stale, and the job dies with "OSError: [Errno 116] Stale file
# handle" on its next print. Jobs 2648172 (all three array tasks) and 2648196
# were lost exactly this way. Logs are pull-only.
#
# Override the target with HOST and DEST:
#   HOST=xsz96350@login23-4.hpc.itc.rwth-aachen.de ./sync_cluster.sh push --go

set -euo pipefail

HOST="${HOST:-xsz96350@login23-1.hpc.itc.rwth-aachen.de}"
DEST="${DEST:-/home/xsz96350/K-RES}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: $0 {push [--go] | pull-logs}" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage

case "$1" in
  push)
    dry=(--dry-run)
    if [[ "${2:-}" == "--go" ]]; then
      dry=()
    else
      echo "dry run; add --go to apply" >&2
    fi
    rsync -av "${dry[@]}" --delete \
      --exclude-from="${REPO}/.rsync-exclude" \
      "${REPO}/" "${HOST}:${DEST}/"
    ;;
  pull-logs)
    rsync -av --progress \
      "${HOST}:${DEST}/onereplay/slurm/outputs/" \
      "${REPO}/onereplay/slurm/outputs/"
    ;;
  *)
    usage
    ;;
esac
