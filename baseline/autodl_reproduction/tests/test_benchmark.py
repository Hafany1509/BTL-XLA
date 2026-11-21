import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('npr_benchmark', Path(__file__).parents[1] / 'test.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_raw_wrapped_and_parallel_weights(self):
        value = object()
        for state in ({'conv.weight': value}, {'model': {'module.conv.weight': value}},
                      {'state_dict': {'module.conv.weight': value}}):
            weights = benchmark.checkpoint_weights(state)
            self.assertEqual(list(weights), ['conv.weight'])
            self.assertIs(weights['conv.weight'], value)

    def test_preserves_nonprefix_module_name(self):
        self.assertEqual(list(benchmark.checkpoint_weights({'modulex.weight': 1})), ['modulex.weight'])

    def test_nested_and_flat_layouts_and_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generator in benchmark.GENERATORS:
                categories = ['car', 'cat'] if generator in ('progan', 'stylegan', 'stylegan2', 'cyclegan') else ['']
                for category in categories:
                    for label in ('0_real', '1_fake'):
                        (root / generator / category / label).mkdir(parents=True)
            benchmark.check_test_layout(root)
            (root / 'stargan' / '1_fake').rmdir()
            with self.assertRaisesRegex(ValueError, 'exactly 0_real and 1_fake'):
                benchmark.check_test_layout(root)


if __name__ == '__main__':
    unittest.main()
