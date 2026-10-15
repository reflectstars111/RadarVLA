#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
paths=${RADARVLA_PATHS_CONFIG:-"$project_root/configs/local_paths.json"}
[[ -f "$paths" ]] || { printf 'Missing local path configuration: %s\n' "$paths" >&2; exit 2; }
python=$(/usr/bin/python3 - "$paths" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['python'])
PY
)
[[ -x "$python" ]] || { printf 'Python is not executable: %s\n' "$python" >&2; exit 2; }
exec "$python" "$project_root/scripts/radarvla_cli.py" --project-root "$project_root" --paths "$paths" "$@"
