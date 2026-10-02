#!/usr/bin/env bash
# Download the labelled class dataset published by liansecurityOS/apk-obfucation-detection.
# The upstream repository has no license file, so the data is fetched on demand rather than
# vendored here. Check the upstream terms before redistributing a model trained on it.
set -euo pipefail

out="${1:-data/new_train.csv}"
mkdir -p "$(dirname "$out")"
curl -fsSL -o "$out" \
  https://raw.githubusercontent.com/liansecurityOS/apk-obfucation-detection/main/new_train.csv
echo "wrote $out ($(wc -l < "$out") lines)"
