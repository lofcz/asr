import unittest
import numpy as np
from runtime import pcm_crop


class CropTests(unittest.TestCase):
    def test_source_sample_timeline(self):
        x = np.arange(100, dtype=np.float32)
        np.testing.assert_array_equal(pcm_crop(x, 2., 3., sample_rate=10), x[20:50])

    def test_single_sample_resampling_rounding_at_eof_is_not_padded(self):
        x = np.arange(99, dtype=np.float32)
        np.testing.assert_array_equal(pcm_crop(x, 7., 3., sample_rate=10), x[70:])

    def test_real_truncation_and_invalid_intervals_fail(self):
        x = np.arange(95, dtype=np.float32)
        for start, duration in [(7., 3.), (-1., 1.), (0., 0.)]:
            with self.assertRaises(ValueError): pcm_crop(x, start, duration, sample_rate=10)

    def test_nonfinite_samples_fail(self):
        with self.assertRaises(ValueError): pcm_crop(np.array([np.nan]), 0, 1, sample_rate=1)


if __name__ == '__main__': unittest.main()
