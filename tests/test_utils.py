import unittest
import sys, os

import numpy as np

import eodfundeq.utils


class UtilsTest(unittest.TestCase):
    def test_ffill(self):
        """Test that ffill is working as expected in a standard use case."""
        print(f"\nRunning test method {self._testMethodName}\n")
        arr = np.array([
            [1, 2, np.nan],
            [3, np.nan, 4], 
            [np.nan, np.nan, 5],
            [6, 7, np.nan],
            [8, np.nan, np.nan]
        ])

        arr_ffill = np.array([
            [1, 2, np.nan],
            [3, 2, 4],
            [3, 2, 5],
            [6, 7, np.nan],
            [8, np.nan, np.nan]
        ])

        self.assertTrue(np.allclose(eodfundeq.utils.ffill(arr), arr_ffill, equal_nan=True))

    def test_rolling_sum_basic(self):
        """Test the rolling sum on a numpy array"""
        n_periods = 3
        arr = np.array([
            [ 0,  1,  2],
            [ 3,  5,  7],
            [ 9, 12, 15],
            [18, 22, 26],
            [30, 35, 40],
        ], dtype=float)

        actual_rolling_sum = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [12., 18., 24.],
            [30., 39., 48.],
            [57., 69., 81.],
        ])

        self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods),
                        actual_rolling_sum, 
                        equal_nan=True))

    def test_rolling_sum_min_obs(self):
        """Test the rolling sum on a numpy array"""
        n_periods = 3
        arr = np.array([
            [ 0,  1,  2],
            [ 3,  5,  np.nan],
            [ 9, 12, 15],
            [18, 22, 26],
            [30, 35, 40],
        ], dtype=float)

        actual_rolling_sum = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [12., 18., 17.],
            [30., 39., 41.],
            [57., 69., 81.],
        ])

        with self.subTest(min_obs=2):
            self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods, min_obs=2),
                            actual_rolling_sum, 
                            equal_nan=True))

        actual_rolling_sum[1:4, 2] = np.nan
        with self.subTest(min_obs=3):
            self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods, min_obs=3),
                            actual_rolling_sum, 
                            equal_nan=True))


if __name__ == '__main__':
    unittest.main()