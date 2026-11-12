"""Create paired clean and JPEG evaluation manifests for NPR."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from PIL import Image


FIELDS = ("dataset", "image_path", "true_label", "sha256", "image_id")
QUALITIES = (90, 70, 50)


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain: {', '.join(FIELDS)}")
        rows = []
        seen_ids = set()
        for row in reader:
            if any(row.get(field, "").strip().lower() in {"false", "0"} for field in ("eligible", "valid")):
                continue
            if (
                not all(row.get(field) for field in FIELDS)
                or row["true_label"] not in {"0", "1"}
                or row["image_id"] in seen_ids
            ):
                raise ValueError(f"Invalid or duplicate manifest record on line {reader.line_num}")
            seen_ids.add(row["image_id"])
            rows.append({field: row[field] for field in FIELDS})
    if not rows:
        raise ValueError("The manifest contains no eligible images")
    return rows


def sample_rows(rows, limit, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["dataset"], row["true_label"]].append(index)
    rng = random.Random(seed)
    selected = {
        index
        for key in sorted(groups)
        for index in rng.sample(groups[key], min(limit, len(groups[key])))
    }
    return [row for index, row in enumerate(rows) if index in selected]


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare(rows, image_root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".jpeg-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        variants = {quality: [] for quality in QUALITIES}
        for quality in QUALITIES:
            (stage / f"q{quality}").mkdir(parents=True)

        for index, row in enumerate(rows, start=1):
            source = image_root / row["image_path"]
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"].lower():
                raise ValueError(f"SHA-256 mismatch for source image: {source}")

            with Image.open(io.BytesIO(data)) as original:
                with original.convert("RGB") as rgb:
                    for quality in QUALITIES:
                        relative = Path(f"q{quality}") / f"{index:06d}.jpg"
                        destination = stage / relative
                        rgb.save(destination, "JPEG", quality=quality, subsampling="4:2:0")
                        variants[quality].append({
                            **row,
                            "image_path": relative.as_posix(),
                            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        })

        write_manifest(stage / "clean.csv", rows)
        for quality, records in variants.items():
            write_manifest(stage / f"q{quality}.csv", records)
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude image IDs and file hashes from an earlier evaluation")
    parser.add_argument("--per-group", type=int, help="Sample up to this many images per source and label")
    parser.add_argument("--seed", type=int, default=100, help="Sampling seed")
    args = parser.parse_args()
    if args.per_group is not None and args.per_group < 1:
        parser.error("--per-group must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")

    try:
        rows = read_manifest(args.manifest)
        if args.exclude_manifest is not None:
            excluded = read_manifest(args.exclude_manifest)
            excluded_ids = {row["image_id"] for row in excluded}
            excluded_hashes = {row["sha256"].lower() for row in excluded}
            rows = [
                row for row in rows
                if row["image_id"] not in excluded_ids and row["sha256"].lower() not in excluded_hashes
            ]
            if not rows:
                raise ValueError("The exclusion manifest leaves no eligible images")
        if args.per_group is not None:
            rows = sample_rows(rows, args.per_group, args.seed)
        prepare(rows, args.image_root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"JPEG preparation failed: {error}\n")
    print(f"Prepared {len(rows)} images for each condition in {args.output}")


if __name__ == "__main__":
    main()
