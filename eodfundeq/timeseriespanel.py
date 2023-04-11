import numpy as np
import pandas as pd

from eodfundeq import utils


class TimeSeriesPanel(object):
    def __init__(self, symbols, frequency, start, end='', dtype=np.float32, default_val=None):
        self.dtype = dtype
        if default_val is not None:
            self.default_val = default_val
        elif dtype in (float, np.float32):
            self.default_val = np.nan
        else:
            raise ValueError('Must specify the default value for the data panel.')

        self.symbols = np.array(symbols)
        if frequency in ('b', 'm'):
            self.frequency = frequency
        else:
            raise ValueError(f'Unsupported frequency: {frequency}')

        self.start = pd.Timestamp(start)
        if not end:
            self.end = pd.Timestamp.now().round('d') - self._get_offset(1)
        else:
            self.end = pd.Timestamp(end)

        self.start_str = self.start.strftime('%Y-%m-%d')
        self.end_str = self.end.strftime('%Y-%m-%d')
        self.dates = pd.date_range(self.start, self.end, freq=self.frequency)

        # Initialize cached time series data
        self._data = self._create_data_rows(self.n_dates)
                
        # initialize hash table to look up date locations
        self._date_map = pd.Series({self.dates[j]: j for j in range(len(self.dates))},
            dtype=int)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if new_data.shape != (self.n_dates, self.n_symbols):
            raise ValueError((f'Expected dimensions {(self.n_dates, self.n_symbols)}'
                              f'but received {new_data.shape}'))
        self._data = new_data

    def _create_data_rows(self, n_rows):
        return self.default_val * np.zeros((n_rows, self.n_symbols), dtype=self.dtype)

    def _get_offset(self, n_periods):
        if self.frequency == 'b':
            return pd.tseries.offsets.BusinessDay(n_periods)
        elif self.frequency == 'm':
            return pd.tseries.offsets.MonthEnd(n_periods)
        else:
            raise ValueError(f'Unsupported frequency: {frequency}')

    def _get_loc(self, date):
        try:
            return self._date_map[date]
        except KeyError:
            return self._date_map[pd.Timestamp(date) + self._get_offset(0)]

    def add_time_series(self, input_ts, idx_symbol):
        if not input_ts.size:
            return
        ts = input_ts.loc[(self.start <= input_ts.index) &
                          (input_ts.index <= self.end)]
        if not ts.size:
            return
        # Add each time series' data to the respective panel
        idx_date = self._get_loc(ts.index[0])
        if idx_date < 0:
            raise ValueError(f'Missing date: {ts.index[0]}')
        L = ts.shape[0]
        self._data[idx_date:idx_date+L, idx_symbol] = ts.values

    def set_data(self, new_data):
        self.data = new_data

    def ffill(self):
        # Fill non-trailing NaNs in close and adjusted_close time series
        self._data = utils.ffill(self._data)

    @property
    def n_dates(self):
        return self.dates.size
    
    @property
    def n_symbols(self):
        return len(self.symbols)