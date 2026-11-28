"""Inference for the team's full AutoDL JPEG model; no upstream submodule required."""
from functools import lru_cache
import argparse
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import time

import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / 'models/npr_jpeg_50ep.pth'
EXPECTED_SHA256 = 'd7b07f94411280632e0d2217810cc01ccf0c67650ee44ab98009f92df15f6354'
PREPROCESS_CONFIG = dict(
    rgb=True, exif_transpose=False, resize=[256, 256], interpolation='bilinear',
    crop=False, flip=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    threshold=0.5, decision='fake_score > 0.5',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def architecture():
    # A private module avoids collisions with the legacy demo's `networks` package.
    path = HERE / 'code/networks/resnet.py'
    spec = importlib.util.spec_from_file_location('_team_autodl_jpeg_resnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet50


def read_rgb(image):
    # Match ImageFolder.pil_loader: RGB conversion, no EXIF transpose.
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert('RGB')
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    raise TypeError('image must be a path or a PIL image')


def recompress(image, quality):
    if quality not in (0, 90, 70, 50):
        raise ValueError('jpeg_quality must be 0, 90, 70 or 50')
    rgb = read_rgb(image)
    if quality == 0:
        return rgb
    with BytesIO() as stream:
        rgb.save(stream, format='JPEG', quality=quality, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as compressed:
            return compressed.convert('RGB')


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS_CONFIG['mean'], PREPROCESS_CONFIG['std']),
    ])


class NPRJPEGDetector:
    def __init__(self, checkpoint=CHECKPOINT, threads=4, expected_sha256=EXPECTED_SHA256):
        if threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint)
        if self.checkpoint_sha256 != expected_sha256:
            raise ValueError('Wrong checkpoint for NPR + JPEG AutoDL: expected {}, got {}'.format(
                expected_sha256, self.checkpoint_sha256))
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and isinstance(state.get('model'), dict):
            state = state['model']
        if not isinstance(state, dict) or not state or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
            raise ValueError('Checkpoint must contain a named tensor state dictionary')
        clean = {key.removeprefix('module.'): value for key, value in state.items()}
        if len(clean) != len(state):
            raise ValueError('Duplicate checkpoint keys after removing module. prefix')
        self.model = architecture()(num_classes=1)
        self.model.load_state_dict(clean, strict=True)
        self.model.cpu().eval()
        self.preprocess = build_preprocess()
        self.model_version = 'npr-jpeg-autodl-50ep-sha256:' + self.checkpoint_sha256

    def predict_batch(self, images, jpeg_quality=0):
        if jpeg_quality not in (0, 90, 70, 50):
            raise ValueError('jpeg_quality must be 0, 90, 70 or 50')
        if not images:
            return []
        started = time.perf_counter()
        tensors = torch.stack([self.preprocess(recompress(image, jpeg_quality)) for image in images])
        with torch.inference_mode():
            logits = self.model(tensors).reshape(-1)
            scores = logits.sigmoid()
        if not torch.isfinite(logits).all() or not torch.isfinite(scores).all():
            raise ValueError('NPR produced non-finite predictions')
        elapsed = (time.perf_counter() - started) * 1000 / len(images)
        return [dict(label='FAKE' if score > 0.5 else 'REAL', fake_score=score,
                     raw_logit=logit, inference_time_ms=elapsed,
                     model_version=self.model_version, checkpoint_sha256=self.checkpoint_sha256)
                for logit, score in zip(logits.tolist(), scores.tolist())]

    def predict(self, image, jpeg_quality=0):
        return self.predict_batch([image], jpeg_quality=jpeg_quality)[0]


@lru_cache(maxsize=4)
def load_model(checkpoint=CHECKPOINT, threads=4):
    return NPRJPEGDetector(checkpoint=checkpoint, threads=threads)


def predict(image, model=None, jpeg_quality=0):
    return (model if model is not None else load_model()).predict(image, jpeg_quality=jpeg_quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--jpeg-quality', type=int, default=0, choices=(0, 90, 70, 50))
    args = parser.parse_args()
    detector = load_model(args.checkpoint, args.threads)
    for image in args.images:
        print(json.dumps(dict(image=image, jpeg_quality=args.jpeg_quality,
                              **detector.predict(image, jpeg_quality=args.jpeg_quality))))


if __name__ == '__main__':
    main()
