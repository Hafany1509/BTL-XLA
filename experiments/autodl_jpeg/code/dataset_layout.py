"""Locate extracted CNNDetection splits without moving or modifying images."""
import argparse
import os
from pathlib import Path

PAPER_CLASSES = ('car', 'cat', 'chair', 'horse')
LABELS = {'0_real', '1_fake'}


def check_root(root, classes):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Image directory does not exist: {}'.format(root))
    categories = classes or sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')
    )
    if not categories:
        raise ValueError('No image categories found in {}'.format(root))
    for category in categories:
        path = root / category
        if not path.is_dir():
            raise ValueError('Missing category directory: {}'.format(path))
        names = {p.name for p in path.iterdir() if p.is_dir()}
        if names != LABELS:
            raise ValueError('{} must contain exactly 0_real and 1_fake directories; found {}'.format(
                path, sorted(names)))
    return str(root)


def resolve_split(base, split, classes, explicit=None):
    if explicit:
        return check_root(explicit, classes)
    base = Path(base).expanduser().resolve()
    # A custom --train_split/--val_split is authoritative.
    if split not in ('train', 'val'):
        return check_root(base / split, classes)
    aliases = {split, 'progan_' + split}
    matches = []
    if base.is_dir():
        for current, dirs, _ in os.walk(str(base)):
            current = Path(current)
            depth = len(current.relative_to(base).parts)
            # Do not walk hundreds of thousands of image directories.
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and
                             d not in LABELS and d not in PAPER_CLASSES)
            if depth >= 4:
                dirs[:] = []
            if current.name not in aliases:
                continue
            try:
                matches.append(check_root(current, classes))
            except ValueError:
                pass
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError('Multiple {} splits found: {}. Set --{}_root explicitly.'.format(
            split, ', '.join(matches), split))
    raise ValueError(
        'Cannot find extracted {} images under {}. Extract the archives first '
        '(7z may produce progan_train.zip, which must also be unzipped), '
        'or pass --{}_root /path/to/categories. Expected category/0_real and category/1_fake.'.format(
            split, base, split))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--train_root')
    parser.add_argument('--val_root')
    parser.add_argument('--classes', default=','.join(PAPER_CLASSES))
    args = parser.parse_args()
    classes = [x.strip() for x in args.classes.split(',') if x.strip()]
    try:
        for split in ('train', 'val'):
            print('{}: {}'.format(split, resolve_split(
                args.dataroot, split, classes, getattr(args, split + '_root'))))
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
