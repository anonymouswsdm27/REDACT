#!/usr/bin/env bash
# Unpack the per-sweep unit tarballs so the figure/table scripts can read them.
# Run once from the repo root:  bash scripts/extract_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for f in results/units/*.tar.gz; do
    echo "extracting $f"
    tar xzf "$f" -C results/units/
done
echo "done — unit dirs are now under results/units/<sweep>/"
