#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ENV="$ROOT/.venv-build-linux"
OUT="$ROOT/artifacts/linux-arch-x86_64"
PYINSTALLER_WORK="$ROOT/build/pyinstaller"

cd "$ROOT"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required to build this app")
PY

python3 -m venv --clear "$BUILD_ENV"
"$BUILD_ENV/bin/python" -m pip install --disable-pip-version-check "pyinstaller>=6,<7"
mkdir -p "$OUT" "$PYINSTALLER_WORK/spec"

"$BUILD_ENV/bin/python" -m PyInstaller \
    --noconfirm \
    --clean \
    --onefile \
    --console \
    --name demo5type \
    --paths "$ROOT" \
    --distpath "$OUT" \
    --workpath "$PYINSTALLER_WORK/work" \
    --specpath "$PYINSTALLER_WORK/spec" \
    "$ROOT/scripts/demo5type_cli.py"

"$OUT/demo5type" >/dev/null
sha256sum "$OUT/demo5type" | awk '{print $1 "  demo5type"}' > "$OUT/demo5type.sha256"
printf 'Built: %s\n' "$OUT/demo5type"
cat "$OUT/demo5type.sha256"
