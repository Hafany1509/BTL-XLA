#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
PY="$JPEG_ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd "$JPEG_ROOT/code"
train_args=(--dataroot "$JPEG_DATA" --classes car,cat,chair,horse --batch_size "$JPEG_BATCH_SIZE" --num_threads "$JPEG_WORKERS" --lr 0.0002 --delr_freq 10 --niter 50 --loss_freq 100 --save_latest_freq 2000 --save_epoch_freq 1 --checkpoints_dir "$JPEG_CHECKPOINTS" --skip_test --blur_prob 0 --jpg_prob 0.5 --jpg_qual 50,95 --jpg_method cv2,pil)
case "${1:-}" in
    check)
        "$PY" preview_jpeg.py --train-root "$JPEG_DATA/extracted/train" --output "$JPEG_ROOT/preview"
        "$PY" train.py "${train_args[@]}" --name check-jpeg- --check_data
        ;;
    train)
        if [ -e "$JPEG_CHECKPOINTS/$JPEG_RUN_NAME/opt.txt" ]; then
            echo 'This B run already exists. Use resume, or change JPEG_RUN_NAME in env.sh for a fresh run.'
            exit 1
        fi
        "$PY" train.py "${train_args[@]}" --name "$JPEG_RUN_NAME" --exact_name
        sha256sum "$JPEG_CHECKPOINTS/$JPEG_RUN_NAME/model_epoch_last.pth"
        ;;
    resume)
        "$PY" train.py "${train_args[@]}" --name "$JPEG_RUN_NAME" --exact_name --resume_state "$JPEG_CHECKPOINTS/$JPEG_RUN_NAME/training_state.pth"
        sha256sum "$JPEG_CHECKPOINTS/$JPEG_RUN_NAME/model_epoch_last.pth"
        ;;
    test-A|test-B)
        model="$JPEG_BASELINE"
        if [ "$1" = test-B ]; then model="$JPEG_CHECKPOINTS/$JPEG_RUN_NAME/model_epoch_last.pth"; fi
        test -f "$model" || { echo "Missing model: $model"; exit 1; }
        mkdir -p "$JPEG_ROOT/results"
        for quality in 0 90 70 50; do
            condition="q$quality"
            if [ "$quality" = 0 ]; then condition=original; fi
            "$PY" test.py --dataroot "$JPEG_TEST" --model_path "$model" --batch_size "$JPEG_BATCH_SIZE" --num_threads "$JPEG_WORKERS" \
                --test_jpeg_quality "$quality" --skip_completed --results_csv "$JPEG_ROOT/results/${1#test-}_${condition}.csv"
        done
        ;;
    summary)
        "$PY" "$JPEG_ROOT/summarize.py" --results "$JPEG_ROOT/results"
        ;;
    *)
        echo 'Use: bash run.sh check|test-A|train|resume|test-B|summary'
        exit 1
        ;;
esac
