"""Verify JPEG behavior on real training images and save a small preview."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS

from data.datasets import data_augment, jpeg_from_key, jpeg_in_memory, prepare_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for category in ('car', 'cat', 'chair', 'horse'):
        for label in ('0_real', '1_fake'):
            folder = Path(args.train_root) / category / label
            samples = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTENSIONS)
            if not samples:
                raise ValueError('Missing training images: ' + str(folder))
            with Image.open(samples[0]) as image:
                original = image.convert('RGB')
            pixels = np.asarray(original)
            prefix = '{}-{}'.format(category, label)
            original.save(output / (prefix + '-original.png'))
            # Train options have JPEG enabled, but clean validation must bypass it.
            options = SimpleNamespace(isTrain=False, jpg_prob=1.0, blur_prob=0,
                                      jpg_qual=[50], jpg_method=['pil'], test_jpeg_quality=0)
            assert np.array_equal(np.asarray(data_augment(original, options)), pixels)
            assert np.array_equal(np.asarray(prepare_image(original, options)), pixels)
            for method in ('pil', 'cv2'):
                for quality in (90, 70, 50):
                    result = jpeg_from_key(pixels, quality, method)
                    assert result.shape == pixels.shape
                    Image.fromarray(result).save(output / (prefix + '-{}-q{}.png'.format(method, quality)))
            options.isTrain = True
            assert np.array_equal(np.asarray(data_augment(original, options)),
                                  jpeg_from_key(pixels, 50, 'pil'))
            options.isTrain = False
            options.test_jpeg_quality = 90
            assert np.array_equal(np.asarray(prepare_image(original, options)),
                                  np.asarray(jpeg_in_memory(original, 90)))
            checked += 1
    print('JPEG CHECK PASSED:', checked, 'real/fake source images, both encoders, clean validation, fixed test compression.')
    print('Preview:', output)


if __name__ == '__main__':
    main()
