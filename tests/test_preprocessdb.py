"""Unit tests for the preprocessdb module."""

import pandas as pd
import numpy as np
import os
import string
import tempfile
import unittest

from eodfundeq.preprocessdb import PreprocessedDataPanel, PreprocessedDBHelper


class UtilsTest(unittest.TestCase):
    """A unit test class for the PreprocessedDBHelper class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = []
        self.datatype_freq_map = {'volume': 'm', 'close': 'm', 'volatility': 'b'}
        self.tmp_path = tempfile.TemporaryDirectory().name
        self.rand_state = np.random.RandomState(seed=123)
        self.n_data_panels = 3
        self.db_helper = PreprocessedDBHelper(self.tmp_path)
        self._load_data_list()
        self._save_data_to_temp_dir()

    def _create_random_df(self, freq, rand_state):
        """Create a random DataFrame."""
        data_start = pd.Timestamp('1990-12-31')
        rand1 = rand_state.choice(24)
        rand2 = 10 + rand_state.choice(24)
        if freq == 'b':
            start = data_start + pd.tseries.offsets.BusinessDay(rand1)
            end = start + pd.tseries.offsets.BusinessDay(rand2)
        elif freq == 'm':
            start = data_start + pd.tseries.offsets.MonthEnd(rand1)
            end = start + pd.tseries.offsets.MonthEnd(rand2)

        dates = pd.date_range(start, end, freq=freq)
        letters = list(string.ascii_uppercase)
        n_symbols = 1 + rand_state.choice(8)
        symbols = rand_state.choice(letters, n_symbols, replace=False)
        return pd.DataFrame(rand_state.normal(size=(len(dates), len(symbols))),
                    index=dates, columns=symbols)

    def _load_data_list(self):
        for _ in range(self.n_data_panels):
            for datatype, freq in self.datatype_freq_map.items():
                datapanel = self._create_random_df(freq=freq, rand_state=self.rand_state)
                self.data_list.append({'data': datapanel, 'frequency': freq, 'datatype': datatype})

        # Create one extra volatility panel
        datapanel = self._create_random_df(freq='b', rand_state=self.rand_state)
        self.data_list.append({'data': datapanel, 'frequency': 'b', 'datatype': 'volatility'})

    def _save_data_to_temp_dir(self):
        """Save the data panels to the directory."""
        for kwargs in self.data_list:
            self.db_helper.save_data(**kwargs)

    def test_save(self):
        """Test that the data panels have been written, as expected."""
        df_meta = self.db_helper.get_metadata()
        self.assertEqual(df_meta.shape[0], len(self.data_list))

    def test_target_version(self):
        """Test that the correct version is retrieved."""
        datatype = 'volatility'
        version = 1
        df1 = self.db_helper.get_panel_data(datatype=datatype, version=version)
        target_data_list = [x for x in self.data_list if x['datatype'] == datatype]
        df2 = target_data_list[version]['data']
        self.assertTrue(df1.equals(df2))

    def test_last_version(self):
        """Test that the last version is retrieved if none is specified."""
        datatype = 'close'
        version = -1
        df1 = self.db_helper.get_panel_data(datatype=datatype)
        target_data_list = [x for x in self.data_list if x['datatype'] == datatype]
        df2 = target_data_list[version]['data']
        self.assertTrue(df1.equals(df2))

    def test_number_of_versions(self):
        """Test that the number of versions for each datatype is as expected."""
        with self.subTest(datatype='volatility'):
            volatility_data_list = [x for x in self.data_list if x['datatype'] == 'volatility']
            self.assertEqual(len(volatility_data_list), 4)

        with self.subTest(datatype='close'):
            close_data_list = [x for x in self.data_list if x['datatype'] == 'close']
            self.assertEqual(len(close_data_list), 3)

    def test_version_numbers(self):
        """Test that the version numbers match our expectation."""
        df_meta = self.db_helper.get_metadata()
        with self.subTest(datatype='volatility'):
            df_vol = df_meta.query('datatype == "volatility"')
            self.assertTrue(np.all(np.arange(0, 4) == df_vol.version.values))

        with self.subTest(datatype='close'):
            df_close = df_meta.query('datatype == "close"')
            self.assertTrue(np.all(np.arange(0, 3) == df_close.version.values))


if __name__ == '__main__':
    unittest.main()
