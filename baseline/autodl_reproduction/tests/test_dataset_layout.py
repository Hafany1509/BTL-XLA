import tempfile
import unittest
from pathlib import Path

from dataset_layout import PAPER_CLASSES, check_root, resolve_split


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def make_split(self, path):
        for category in PAPER_CLASSES:
            for label in ('0_real', '1_fake'):
                (path / category / label).mkdir(parents=True)
        return str(path.resolve())

    def test_original_layout(self):
        train = self.make_split(self.base / 'train')
        val = self.make_split(self.base / 'val')
        self.assertEqual(resolve_split(self.base, 'train', PAPER_CLASSES), train)
        self.assertEqual(resolve_split(self.base, 'val', PAPER_CLASSES), val)

    def test_huggingface_wrappers(self):
        train = self.make_split(self.base / 'extracted' / 'train' / 'progan_train')
        val = self.make_split(self.base / 'extracted' / 'val' / 'progan_val')
        self.assertEqual(resolve_split(self.base, 'train', PAPER_CLASSES), train)
        self.assertEqual(resolve_split(self.base, 'val', PAPER_CLASSES), val)

    def test_explicit_nonstandard_path(self):
        train = self.make_split(self.base / 'custom-images')
        self.assertEqual(resolve_split(self.base, 'train', PAPER_CLASSES, train), train)

    def test_ambiguous_layout_is_rejected(self):
        self.make_split(self.base / 'train')
        self.make_split(self.base / 'progan_train')
        with self.assertRaisesRegex(ValueError, 'Multiple train splits'):
            resolve_split(self.base, 'train', PAPER_CLASSES)

    def test_validation_cannot_fall_back_to_train(self):
        self.make_split(self.base / 'train')
        with self.assertRaisesRegex(ValueError, 'Cannot find extracted val'):
            resolve_split(self.base, 'val', PAPER_CLASSES)

    def test_missing_category_and_extra_label(self):
        root = Path(self.make_split(self.base / 'train'))
        (root / 'car' / '2_other').mkdir()
        with self.assertRaisesRegex(ValueError, 'exactly 0_real and 1_fake'):
            check_root(root, PAPER_CLASSES)
        with self.assertRaisesRegex(ValueError, 'Missing category'):
            check_root(root, ['absent'])

    def test_unextracted_archive_message(self):
        (self.base / 'progan_train.zip').touch()
        with self.assertRaisesRegex(ValueError, 'must also be unzipped'):
            resolve_split(self.base, 'train', PAPER_CLASSES)

    def test_custom_split_and_class_discovery(self):
        custom = self.make_split(self.base / 'my-training-split')
        self.assertEqual(resolve_split(self.base, 'my-training-split', []), custom)


if __name__ == '__main__':
    unittest.main()
