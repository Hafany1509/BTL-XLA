"""CPU inference for the original NPR network. No training or external downloads."""
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
# Author code and checkpoint come from the upstream/npr submodule of the team repo.
UPSTREAM = ROOT.parent / 'upstream' / 'npr'
if not (UPSTREAM / 'networks' / 'resnet.py').is_file():
    raise ImportError('NPR source missing. Run: git submodule update --init --recursive')
sys.path.insert(0, str(UPSTREAM))
from networks.resnet import resnet50  # noqa: E402

CHECKPOINT = UPSTREAM / 'NPR.pth'
PREPROCESS_CONFIG = dict(resize=[256, 256], interpolation='bilinear', crop=False,
                         flip=False, rgb=True, exif_transpose=False,
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(image):
    # Matches torchvision ImageFolder.pil_loader: no EXIF orientation transform.
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert('RGB')
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    raise TypeError('image must be a filesystem path or PIL.Image.Image')


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS_CONFIG['mean'], PREPROCESS_CONFIG['std']),
    ])


class NPRDetector:
    def __init__(self, checkpoint=CHECKPOINT, threads=4):
        torch.set_num_threads(threads)
        if threads < 1:
            raise ValueError('threads must be positive')
        self.threads = threads
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint)
        checkpoint_data = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint_data, dict):
            raise ValueError('Expected a wrapped or flat state dictionary')
        wrapped = isinstance(checkpoint_data.get('model'), dict)
        source_state = checkpoint_data['model'] if wrapped else checkpoint_data
        if not source_state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in source_state.items()):
            raise ValueError('State dictionary must contain named tensors (wrapper key: model)')
        self.checkpoint_format = 'wrapped_model' if wrapped else 'flat_state_dict'
        self.model_version = 'npr-sha256:' + self.checkpoint_sha256
        state = {k.removeprefix('module.'): v for k, v in source_state.items()}
        if len(state) != len(source_state):
            raise ValueError('Duplicate state keys after removing module. prefix')
        self.model = resnet50(num_classes=1)
        self.load_result = self.model.load_state_dict(state, strict=True)
        self.model.cpu().eval()
        self.preprocess = build_preprocess()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.state_entries = len(state)

    def predict_batch(self, images):
        if not images:
            return []
        started = time.perf_counter()
        tensor = torch.stack([self.preprocess(read_rgb(image)) for image in images])
        with torch.inference_mode():
            logits = self.model(tensor).reshape(-1)
            scores = logits.sigmoid()
        if not torch.isfinite(logits).all() or not torch.isfinite(scores).all():
            raise ValueError('Non-finite NPR output')
        elapsed = (time.perf_counter() - started) * 1000 / len(images)
        return [dict(label='FAKE' if score > 0.5 else 'REAL', raw_logit=float(logit),
                     fake_score=float(score), inference_time_ms=elapsed,
                     model_version=self.model_version, checkpoint_sha256=self.checkpoint_sha256)
                for logit, score in zip(logits.tolist(), scores.tolist())]

    def predict(self, image):
        return self.predict_batch([image])[0]


@lru_cache(maxsize=4)
def load_model(checkpoint=CHECKPOINT, threads=4):
    """Reuse one detector for successive requests with the same arguments."""
    return NPRDetector(checkpoint, threads)


def predict(image, model=None):
    """model is an NPRDetector returned by load_model(), or a checkpoint path."""
    if model is None:
        model = load_model()
    elif isinstance(model, (str, Path)):
        model = load_model(Path(model).resolve())
    if not isinstance(model, NPRDetector):
        raise TypeError('model must be an NPRDetector from load_model(), or a checkpoint path')
    return model.predict(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='+')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    detector = load_model(args.checkpoint, args.threads)
    for path in args.image:
        print(json.dumps(dict(image_path=path, **detector.predict(path))))
