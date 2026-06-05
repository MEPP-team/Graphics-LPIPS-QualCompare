import unittest

import numpy as np

from ssim import _automatic_downsample, _matlab_round_positive, ssim


class TestSsim(unittest.TestCase):
    def test_identical_images_score_one(self):
        image = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)

        score, ssim_map = ssim(image, image)

        self.assertAlmostEqual(score, 1.0, places=14)
        self.assertEqual(ssim_map.shape, (246, 246))

    def test_regression_score(self):
        y, x = np.mgrid[:960, :960]
        reference = ((x + 2 * y) % 256).astype(np.uint8)
        distorted = np.clip(reference.astype(np.int16) + ((x % 9) - 4), 0, 255).astype(np.uint8)

        score, ssim_map = ssim(reference, distorted)

        self.assertAlmostEqual(score, 0.995924911649368, places=12)
        self.assertAlmostEqual(score, float(np.mean(ssim_map)), places=14)
        self.assertEqual(ssim_map.shape, (230, 230))

    def test_matlab_rounding_is_used_for_downsampling(self):
        self.assertEqual(_matlab_round_positive(2.5), 3)
        image = np.zeros((640, 640), dtype=np.uint8)
        self.assertEqual(_automatic_downsample(image).shape, (214, 214))

    def test_rejects_custom_window(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Custom SSIM windows"):
            ssim(image, image, window=np.ones((5, 5)))


if __name__ == "__main__":
    unittest.main()
