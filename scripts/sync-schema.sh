#!/usr/bin/env bash
# Copies the llumiverse JSON Schema document into schema/ and regenerates package wire.
#
# Usage:
#   scripts/sync-schema.sh <path to a llumiverse checkout>
#   scripts/sync-schema.sh <commit, tag or branch of vertesia/llumiverse>
#
# schema/SOURCE records the llumiverse commit the copy came from.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
source_arg="${1:?usage: scripts/sync-schema.sh <llumiverse checkout path | git ref>}"
document="common/json-schema/llumiverse.schema.json"
target="$root/schema/llumiverse.schema.json"

if [[ -d "$source_arg" ]]; then
  cp "$source_arg/$document" "$target"
  commit="$(git -C "$source_arg" rev-parse HEAD)"
  if [[ -n "$(git -C "$source_arg" status --porcelain -- "$document")" ]]; then
    commit="$commit (with uncommitted changes)"
  fi
else
  commit="$(git ls-remote https://github.com/vertesia/llumiverse.git "$source_arg" | awk 'NR == 1 { print $1 }')"
  commit="${commit:-$source_arg}"
  curl --fail --silent --show-error --location \
    "https://raw.githubusercontent.com/vertesia/llumiverse/$commit/$document" -o "$target"
fi

printf 'vertesia/llumiverse@%s\n' "$commit" > "$root/schema/SOURCE"
(cd "$root" && go generate ./wire)
echo "schema/llumiverse.schema.json synced from vertesia/llumiverse@$commit"
