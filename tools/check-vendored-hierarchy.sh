#!/usr/bin/env bash
# _rotobot_hierarchy/ is a byte-for-byte copy of rotobot-nuke's reader and
# hierarchy modules (and its LICENSE) at the commit in VENDORED.txt. A copy
# nobody compares drifts; this compares it. Same check as Rotobot-Next's
# packaging/ci/check-vendored-hierarchy.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=_rotobot_hierarchy
sha=$(awk '$1=="commit"{print $2}' "$D/VENDORED.txt")
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: $D/VENDORED.txt has no full commit sha" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
for src in src/rotobot_nuke/reader.py src/rotobot_nuke/hierarchy.py LICENSE; do
  name=$(basename "$src")
  python3 -c 'import sys, urllib.request; open(sys.argv[2], "wb").write(urllib.request.urlopen(sys.argv[1], timeout=60).read())' \
    "https://raw.githubusercontent.com/samhodge-tokgan/rotobot-nuke/$sha/$src" "$tmp/$name"
  cmp -s "$tmp/$name" "$D/$name" || { echo "ERROR: $D/$name differs from rotobot-nuke@${sha:0:7}:$src" >&2; fail=1; }
done
[ $fail = 0 ] && echo "_rotobot_hierarchy matches rotobot-nuke@${sha:0:7}"
exit $fail
