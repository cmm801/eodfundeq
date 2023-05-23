"""Unit tests for the utils module."""

import unittest
import sys, os

import numpy as np

import eodfundeq.utils


class UtilsTest(unittest.TestCase):
    def test_ffill(self):
        """Test that ffill is working as expected in a standard use case."""
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

        true_rolling_sum = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [12., 18., 24.],
            [30., 39., 48.],
            [57., 69., 81.],
        ])

        self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods),
                        true_rolling_sum, 
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

        true_rolling_sum = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [12., 18., 17.],
            [30., 39., 41.],
            [57., 69., 81.],
        ])
        
        with self.subTest(min_obs=2):
            true_rolling_sum_2 = true_rolling_sum.copy()
            true_rolling_sum_2[1,:] = np.sum(arr[:2,:], axis=0)
            self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods, min_obs=2),
                            true_rolling_sum_2, 
                            equal_nan=True))
        
        with self.subTest(min_obs=1):
            true_rolling_sum_1 = true_rolling_sum.copy()
            true_rolling_sum_1[0,:] = arr[0,:]
            true_rolling_sum_1[1,:] = np.nansum(arr[:2,:], axis=0)
            self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods, min_obs=1),
                            true_rolling_sum_1, 
                            equal_nan=True))                            

        true_rolling_sum[1:4, 2] = np.nan
        with self.subTest(min_obs=3):
            self.assertTrue(np.allclose(eodfundeq.utils.rolling_sum(arr, n_periods, min_obs=3),
                            true_rolling_sum, 
                            equal_nan=True))

    def test_rolling_mean_basic(self):
        n_periods = 3
        arr = np.array([
            [ 0,  1,  2],
            [ 3,  5,  7],
            [ 9, 12, 15],
            [18, 22, 26],
            [30, 35, 40],
        ], dtype=float)

        true_rolling_mean = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [4., 6., 8.],
            [10., 13., 16.],
            [19., 23., 27.],
        ])
        self.assertTrue(np.allclose(eodfundeq.utils.rolling_mean(arr, n_periods),
                        true_rolling_mean, 
                        equal_nan=True))

    def test_rolling_mean_min_obs(self):
        n_periods = 3
        min_obs = 2
        arr = np.array([
            [ 0,  1,  2],
            [ 3,  np.nan,  7],
            [ 9, 12, 15],
            [18, 22, 26],
            [30, 35, np.nan],
        ], dtype=float)

        true_rolling_mean = np.array([
            [np.nan, np.nan, np.nan],
            [3/2, np.nan, 9/2],
            [4., 13/2, 8.],
            [10., 34/2., 16.],
            [19., 23., 41/2],
        ])
        self.assertTrue(np.allclose(eodfundeq.utils.rolling_mean(arr, n_periods, min_obs=min_obs),
                        true_rolling_mean, 
                        equal_nan=True))

    def test_rolling_std_basic(self):
        n_periods = 3
        min_obs = 2
        arr = np.array([
            [ 0,  1,  2],
            [ 3,  5,  np.nan],
            [ 9, 12, 15],
            [18, 22, 26],
            [30, 35, 40],
            [32, 13, 24],    
        ], dtype=float)

        true_rolling_std = np.nan * np.ones_like(arr)
        for col in range(arr.shape[1]):
            j = 0
            while j < arr.shape[0]:
                if j < min_obs - 1:
                    j += 1
                    continue
                
                s = slice(max(0, j - n_periods + 1), j+1)
                if (~np.isnan(arr[s, col])).sum() >= min_obs:
                    true_rolling_std[j, col] = np.nanstd(arr[s, col])
                j += 1

        self.assertTrue(np.allclose(eodfundeq.utils.rolling_std(arr, n_periods, min_obs=min_obs),
                        true_rolling_std, 
                        equal_nan=True))



if __name__ == '__main__':
    unittest.main()