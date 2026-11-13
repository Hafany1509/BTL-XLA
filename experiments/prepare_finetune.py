"""Build train and validation pilots from the CNNDetection ProGAN validation ZIP."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from prepare_jpeg import read_manifest, write_manifest


LABELS = {"0_real": "0", "1_fake": "1"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    used = {row["sha256"].lower() for row in read_manifest(args.test_manifest)}
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive, TemporaryDirectory(dir=args.output.parent) as temporary:
        groups = defaultdict(list)
        for item in archive.infolist():
            parts = PurePosixPath(item.filename).parts
            if (
                len(parts) >= 3
                and parts[-3] in args.categories
                and parts[-2] in LABELS
                and PurePosixPath(item.filename).suffix.lower() in IMAGE_EXTENSIONS
            ):
                groups[parts[-3], parts[-2]].append(item)

        stage = Path(temporary) / "prepared"
        records = {"train": [], "val": []}
        skipped = 0
        for category in args.categories:
            for folder, label in LABELS.items():
                candidates = sorted(groups[category, folder], key=lambda item: item.filename)
                rng.shuffle(candidates)
                kept = 0
                for item in candidates:
                    data = archive.read(item)
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in used:
                        skipped += 1
                        continue
                    used.add(digest)
                    split = "train" if kept < args.train_per_group else "val"
                    extension = PurePosixPath(item.filename).suffix.lower()
                    relative = Path(split, category, folder, f"{kept:05d}{extension}")
                    path = stage / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    records[split].append({
                        "dataset": "cnndetection_progan_val",
                        "image_path": relative.as_posix(),
                        "true_label": label,
                        "sha256": digest,
                        "image_id": f"img-{digest}",
                    })
                    kept += 1
                    if kept == args.train_per_group + args.val_per_group:
                        break
                else:
                    raise ValueError(f"Only {kept} distinct images for {category}/{folder}")

        for split, rows in records.items():
            write_manifest(stage / f"{split}.csv", rows)
        summary = {
            "archive": args.archive.name,
            "archive_sha256": file_sha256(args.archive),
            "test_manifest": str(args.test_manifest.resolve()),
            "categories": args.categories,
            "seed": args.seed,
            "train_images": len(records["train"]),
            "val_images": len(records["val"]),
            "duplicates_skipped": skipped,
        }
        (stage / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["car", "chair"])
    parser.add_argument("--train-per-group", type=int, default=40)
    parser.add_argument("--val-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if min(args.train_per_group, args.val_per_group) < 1:
        parser.error("Both split sizes must be positive")
    if len(set(args.categories)) != len(args.categories) or not all(name.isalnum() for name in args.categories):
        parser.error("Category names must be unique and contain only letters or digits")
    if args.output.exists():
        parser.error("--output must be a new directory")
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
