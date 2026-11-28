#!/usr/bin/env bash
JPEG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JPEG_DATA=/root/autodl-tmp/CNNDetection
JPEG_TEST=/root/autodl-tmp/CNNDetection/extracted/test
JPEG_BASELINE=/root/autodl-tmp/checkpoints/NPR-4class-2026_11_21_05_53_01/model_epoch_last.pth
JPEG_CHECKPOINTS="$JPEG_ROOT/checkpoints"
JPEG_RUN_NAME=B-jpeg
JPEG_BATCH_SIZE=32
JPEG_WORKERS=8
