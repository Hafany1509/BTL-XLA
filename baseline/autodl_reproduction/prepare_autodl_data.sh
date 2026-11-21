#!/usr/bin/env bash
# Extract train/validation only. Existing images/archives are retained.
set -euo pipefail
DATA_ROOT="${1:-/root/autodl-tmp/CNNDetection}"
command -v 7z >/dev/null || { echo 'Install 7z: apt-get install -y p7zip-full'; exit 1; }
command -v unzip >/dev/null || { echo 'Install unzip: apt-get install -y unzip'; exit 1; }
for part in 001 002 003 004 005 006 007; do
  test -f "$DATA_ROOT/progan_train.7z.$part" || { echo "Missing progan_train.7z.$part"; exit 1; }
done
test -f "$DATA_ROOT/progan_val.zip" || { echo 'Missing progan_val.zip'; exit 1; }
mkdir -p "$DATA_ROOT/extracted/train" "$DATA_ROOT/extracted/val"
# The official download script uses TWO stages for training: 7z, then unzip.
7z x -aos "$DATA_ROOT/progan_train.7z.001" "-o$DATA_ROOT/extracted/train"
mapfile -d '' TRAIN_ZIPS < <(find "$DATA_ROOT/extracted/train" -maxdepth 3 -type f -name progan_train.zip -print0)
if (( ${#TRAIN_ZIPS[@]} > 1 )); then
  echo 'Multiple progan_train.zip files found; choose the intended archive manually.'
  exit 1
elif (( ${#TRAIN_ZIPS[@]} == 1 )); then
  unzip -n "${TRAIN_ZIPS[0]}" -d "$DATA_ROOT/extracted/train"
fi
unzip -n "$DATA_ROOT/progan_val.zip" -d "$DATA_ROOT/extracted/val"
python "$(dirname "$0")/dataset_layout.py" --dataroot "$DATA_ROOT/extracted"
