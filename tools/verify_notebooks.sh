#!/bin/bash
# Run every chapter notebook headless in a fresh environment built from requirements.txt.
#   bash tools/verify_notebooks.sh            (from the repo's root)
# Prints PASS or FAIL per notebook; executed copies and logs go to $OUT (default /tmp/lift-notebooks).
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-/tmp/lift-notebooks}
mkdir -p "$OUT"
command -v uv >/dev/null || pip install -q uv
[ -x "$OUT/.venv/bin/python" ] || uv venv -q --python 3.12 "$OUT/.venv"
uv pip install -q --python "$OUT/.venv/bin/python" -r "$root/requirements.txt" nbconvert ipykernel
. "$OUT/.venv/bin/activate"
cd "$root/chapters"
status=0
for nb in *.ipynb; do
  if jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
       --output-dir "$OUT" "$nb" > "$OUT/${nb%.ipynb}.log" 2>&1; then
    echo "PASS $nb"
  else
    echo "FAIL $nb (see $OUT/${nb%.ipynb}.log)"; status=1
  fi
done
exit $status
