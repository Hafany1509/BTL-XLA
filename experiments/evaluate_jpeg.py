"""Run the member 1 evaluator for each checkpoint and JPEG condition."""

import argparse
from pathlib import Path
import sys

import pandas as pd

from prepare_jpeg import QUALITIES, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handover", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.handover.resolve()))
    from evaluate_checkpoint import run_evaluation

    checkpoints = {
        "baseline": args.handover / "NPR.pth",
        "control": args.checkpoints / "npr_control.pth",
        "jpeg": args.checkpoints / "npr_jpeg.pth",
    }
    conditions = ["clean", *(f"q{quality}" for quality in QUALITIES)]
    manifests = {name: args.conditions / f"{name}.csv" for name in conditions}
    clean_rows = read_manifest(manifests["clean"])
    clean_keys = {row["image_id"]: (row["dataset"], row["true_label"]) for row in clean_rows}
    all_metrics = []
    for name, path in manifests.items():
        rows = read_manifest(path)
        keys = {row["image_id"]: (row["dataset"], row["true_label"]) for row in rows}
        if keys != clean_keys:
            raise ValueError(f"Condition {name} differs from clean in image IDs, sources or labels")
        root = args.handover if name == "clean" else args.conditions
        manifest = pd.read_csv(path)
        for model, checkpoint in checkpoints.items():
            print(f"Evaluating {model} on {name}", flush=True)
            _, metrics = run_evaluation(
                manifest, checkpoint=checkpoint, image_root=root, output=args.output / name / model,
                batch_size=16, threads=4,
            )
            if not metrics["Complete"].all():
                raise RuntimeError(f"Missing predictions for {model} on {name}")
            all_metrics.append(metrics.assign(condition=name, model=model))
    pd.concat(all_metrics, ignore_index=True).to_csv(args.output / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
