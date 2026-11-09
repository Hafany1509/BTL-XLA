"""Check that the pinned NPR checkpoint loads and runs on CPU."""

import argparse
from pathlib import Path
import sys

import torch


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "npr"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=UPSTREAM / "NPR.pth")
    args = parser.parse_args()

    if not (UPSTREAM / "networks" / "resnet.py").is_file():
        parser.error("Initialize NPR with: git submodule update --init --recursive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(UPSTREAM))
    from networks.resnet import resnet50

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    model = resnet50(num_classes=1)
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in weights.items()},
        strict=True,
    )

    sample = torch.linspace(-1, 1, steps=3 * 224 * 224).reshape(1, 3, 224, 224)
    with torch.inference_mode():
        logit = model.eval()(sample).item()
    if not torch.isfinite(torch.tensor(logit)).item():
        raise RuntimeError("NPR returned a non-finite logit")

    print(f"OK: {args.checkpoint.name} loaded and ran on CPU (synthetic input)")


if __name__ == "__main__":
    main()
