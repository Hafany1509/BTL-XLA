"""Fine-tune the same NPR checkpoint with and without JPEG augmentation."""

import argparse
import csv
import io
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare_finetune import file_sha256
from prepare_jpeg import QUALITIES, read_manifest


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "npr"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Images(Dataset):
    def __init__(self, rows, root, jpeg_probability, seed):
        self.rows = rows
        self.root = root
        self.jpeg_probability = jpeg_probability
        self.rng = random.Random(seed)
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as source:
            rgb = source.convert("RGB")
        if self.jpeg_probability and self.rng.random() < self.jpeg_probability:
            with io.BytesIO() as buffer:
                rgb.save(buffer, "JPEG", quality=self.rng.choice(QUALITIES), subsampling="4:2:0")
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    rgb = compressed.convert("RGB")
        resized = rgb.resize((256, 256), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div_(255)
        return (tensor - self.mean) / self.std, float(row["true_label"])


def verify_splits(train, val, test, root):
    seen = {row["sha256"].lower() for row in test}
    for split, rows in (("train", train), ("val", val)):
        if {row["true_label"] for row in rows} != {"0", "1"}:
            raise ValueError(f"{split} must contain both labels")
        for row in rows:
            path = root / row["image_path"]
            actual = file_sha256(path)
            if actual != row["sha256"].lower():
                raise ValueError(f"SHA-256 mismatch for {path}")
            if actual in seen:
                raise ValueError(f"Duplicate image within or across splits: {path}")
            seen.add(actual)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += ((logits.detach() > 0) == labels.bool()).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def fit(args, mode, state, train, val, device, resnet50):
    torch.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    train_images = Images(train, args.image_root, 0.5 if mode == "jpeg" else 0.0, args.seed)
    val_images = Images(val, args.image_root, 0.0, args.seed)
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True, generator=shuffle)
    val_loader = DataLoader(val_images, batch_size=args.batch_size)
    model = resnet50(num_classes=1)
    model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    with (args.output / f"{mode}_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"))
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)
            writer.writerow((epoch, train_loss, train_acc, val_loss, val_acc))
            stream.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}},
                           args.output / f"npr_{mode}.pth")
            print(f"{mode} epoch {epoch}: val_loss={val_loss:.6f} val_accuracy={val_acc:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=UPSTREAM / "NPR.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not 0 < args.lr < float("inf"):
        parser.error("Epochs, batch size, threads and learning rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    if not (UPSTREAM / "networks" / "resnet.py").is_file():
        parser.error("Initialize the NPR submodule first")

    train = read_manifest(args.train_manifest)
    val = read_manifest(args.val_manifest)
    verify_splits(train, val, read_manifest(args.test_manifest), args.image_root)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(UPSTREAM))
    from networks.resnet import resnet50

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint.get("model", checkpoint).items()}
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": {split: file_sha256(getattr(args, f"{split}_manifest")) for split in ("train", "val", "test")},
        "train_images": len(train),
        "val_images": len(val),
        "jpeg_probability": 0.5,
        "jpeg_qualities": QUALITIES,
        "selection": "lowest clean validation BCE",
        "torch": torch.__version__,
    }
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    for mode in ("control", "jpeg"):
        fit(args, mode, state, train, val, torch.device(args.device), resnet50)


if __name__ == "__main__":
    main()
