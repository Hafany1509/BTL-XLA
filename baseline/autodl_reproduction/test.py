"""Evaluate the eight ForenSynths generators used in the NPR benchmark."""
import csv
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

GENERATORS = ('progan', 'stylegan', 'stylegan2', 'biggan',
              'cyclegan', 'stargan', 'gaugan', 'deepfake')
DEFAULT_DATA = '/root/autodl-tmp/CNNDetection/extracted/test'


def checkpoint_weights(state):
    """Accept raw, wrapped and DataParallel checkpoints without changing weights."""
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    return OrderedDict((key[7:] if key.startswith('module.') else key, value)
                       for key, value in state.items())


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
            parser.set_defaults(dataroot=DEFAULT_DATA,
                                batch_size=32, num_threads=8)
            parser.add_argument('--results_csv',
                                default='/root/autodl-tmp/npr-test-results.csv')
            return parser

    opt = BenchmarkOptions().parse(print_options=False)
    if not opt.gpu_ids or not torch.cuda.is_available():
        raise SystemExit('This evaluator requires a CUDA GPU (the shared validate.py uses CUDA).')
    if not opt.model_path or not Path(opt.model_path).is_file():
        raise SystemExit('Provide --model_path /path/to/checkpoint.pth (file must exist).')
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

    model = resnet50(num_classes=1)
    state = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(checkpoint_weights(state), strict=True)
    model.cuda().eval()

    output = Path(opt.results_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Checkpoint:', opt.model_path, flush=True)
    print('Test root:', root, flush=True)
    print('Results:', output, flush=True)
    print('Metrics below are percentages; Mean is the unweighted generator mean.', flush=True)
    rows = []
    start = time.perf_counter()
    with output.open('w', newline='', encoding='utf-8') as stream:
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
    print('Mean: Acc {:.2f}% | AP {:.2f}%'.format(mean[0], mean[1]), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
