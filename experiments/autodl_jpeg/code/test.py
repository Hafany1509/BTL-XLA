"""Evaluate the eight ForenSynths generators used in the NPR benchmark."""
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

GENERATORS = ('progan', 'stylegan', 'stylegan2', 'biggan',
              'cyclegan', 'stargan', 'gaugan', 'deepfake')
DEFAULT_DATA = '/root/autodl-tmp/CNNDetection/extracted/test'
DEFAULT_MODEL = ('/root/autodl-tmp/checkpoints/'
                 'NPR-4class-2026_11_21_05_53_01/model_epoch_last.pth')


def check_test_layout(root):
    """Reject missing generators or incorrect binary-label folder layouts."""
    labels = {'0_real', '1_fake'}
    for generator in GENERATORS:
        folder = Path(root) / generator
        if not folder.is_dir():
            raise ValueError('Missing test directory: {}'.format(folder))
        children = {p.name for p in folder.iterdir()
                    if p.is_dir() and not p.name.startswith('.')}
        leaves = [folder] if children & labels else [folder / c for c in sorted(children)]
        if not leaves:
            raise ValueError('Empty generator directory: {}'.format(folder))
        for leaf in leaves:
            names = {p.name for p in leaf.iterdir()
                     if p.is_dir() and not p.name.startswith('.')}
            if names != labels:
                raise ValueError('{} must contain exactly 0_real and 1_fake; found {}'.format(
                    leaf, sorted(names)))


def main():
    import numpy as np
    import torch
    from networks.resnet import resnet50
    from options.test_options import TestOptions
    from validate import validate

    class BenchmarkOptions(TestOptions):
        def initialize(self, parser):
            parser = super().initialize(parser)
            parser.set_defaults(dataroot=DEFAULT_DATA, model_path=DEFAULT_MODEL,
                                batch_size=32, num_threads=8)
            parser.add_argument('--skip_completed', action='store_true')
            parser.add_argument('--results_csv',
                                default='/root/autodl-tmp/npr-test-results.csv')
            return parser

    opt = BenchmarkOptions().parse(print_options=False)
    if opt.gpu_ids and not torch.cuda.is_available():
        raise SystemExit('CUDA GPU is unavailable. Use --gpu_ids -1 for CPU evaluation.')
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else 'cpu')
    if not Path(opt.model_path).is_file():
        raise SystemExit('Checkpoint not found: ' + opt.model_path)
    try:
        check_test_layout(opt.dataroot)
    except ValueError as exc:
        raise SystemExit(str(exc))

    random.seed(100)
    os.environ['PYTHONHASHSEED'] = '100'
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # Match the original ForenSynths test configuration: resize to loadSize,
    # no crop; discover all categories, including the full ProGAN test set.
    opt.isTrain = False
    opt.no_resize = False
    opt.no_crop = True
    opt.serial_batches = True
    opt.class_bal = False
    opt.classes = []
    root = opt.dataroot

    output = Path(opt.results_csv)
    metadata_path = output.with_suffix('.json')
    digest = hashlib.sha256()
    with Path(opt.model_path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    input_digest = hashlib.sha256()
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp'}
    for generator in GENERATORS:
        for image in sorted((Path(root) / generator).rglob('*')):
            if image.is_file() and image.suffix.lower() in extensions:
                stat = image.stat()
                record = (image.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns)
                input_digest.update(json.dumps(record).encode('utf-8'))
    import PIL
    import torchvision
    metadata = dict(checkpoint_sha256=digest.hexdigest(),
                    input_fingerprint=input_digest.hexdigest(),
                    test_root=str(Path(root).resolve()),
                    test_jpeg_quality=opt.test_jpeg_quality, subsampling='4:2:0',
                    compression_position='before resize at native resolution',
                    threshold=0.5, resize=[opt.loadSize, opt.loadSize], crop=False,
                    torch=torch.__version__, torchvision=torchvision.__version__,
                    pillow=PIL.__version__)
    if opt.skip_completed and output.exists() and metadata_path.exists():
        with output.open(newline='', encoding='utf-8') as handle:
            existing = list(csv.DictReader(handle))
        names = [row['generator'] for row in existing]
        if names == list(GENERATORS) + ['Mean'] and json.loads(metadata_path.read_text()) == metadata:
            print('Already complete:', output, flush=True)
            return
    model = resnet50(num_classes=1)
    state = torch.load(opt.model_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    output = Path(opt.results_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Checkpoint:', opt.model_path, flush=True)
    print('Test root:', root, flush=True)
    print('Results:', output, flush=True)
    print('Metrics below are percentages; Mean is the unweighted generator mean.', flush=True)
    rows = []
    start = time.perf_counter()
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['generator', 'images', 'accuracy_percent', 'ap_percent',
                         'real_accuracy_percent', 'fake_accuracy_percent', 'seconds'])
        for index, generator in enumerate(GENERATORS, 1):
            opt.dataroot = str(Path(root) / generator)
            print('[{}/{}] Testing {} ...'.format(index, len(GENERATORS), generator), flush=True)
            tick = time.perf_counter()
            acc, ap, real_acc, fake_acc, truth, _ = validate(model, opt)
            elapsed = time.perf_counter() - tick
            metrics = [100 * value for value in (acc, ap, real_acc, fake_acc)]
            rows.append(metrics)
            writer.writerow([generator, len(truth)] + metrics + [round(elapsed, 2)])
            stream.flush()
            print('{}: {} images | Acc {:.2f}% | AP {:.2f}% | {:.1f}s'.format(
                generator, len(truth), metrics[0], metrics[1], elapsed), flush=True)
        mean = np.mean(rows, axis=0)
        writer.writerow(['Mean', ''] + mean.tolist() + [round(time.perf_counter() - start, 2)])
    os.replace(temporary, output)
    temporary_metadata = metadata_path.with_name(metadata_path.name + '.tmp')
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    os.replace(temporary_metadata, metadata_path)
    print('Mean: Acc {:.2f}% | AP {:.2f}%'.format(mean[0], mean[1]), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
