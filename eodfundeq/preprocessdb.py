"""This module provides classes to help manage preprocessed time series panel data.
"""

import fastparquet
import numpy as np
import os
import pandas as pd


class PreprocessedDBHelper(object):
    """Class to help manage preprocessed time series panel data."""
    def __init__(self, base_path):
        self.base_path = base_path

    def get_metadata_filename(self):
        """Get name of file containing the metadata for all time series panels."""
        return os.path.join(self.base_path, 'metadata.csv')

    def get_datatype_path(self, datatype=''):   # pylint: disable=unused-argument
        """Get path to file containing time series panel for a specific data type."""
        return self.base_path

    def get_metadata(self):
        """Get a DataFrame containing metadata for all time series panels."""
        filename = self.get_metadata_filename()
        if not os.path.isfile(filename):
            return pd.DataFrame([])
        else:
            return pd.read_csv(filename)

    def append_metadata(self, dp_obj):
        """Update the meta data file that keeps track of all saved data panels."""
        base_data = dict(
            datatype=dp_obj.datatype,
            start=dp_obj.start,
            end=dp_obj.end,
            frequency=dp_obj.frequency,
            relative_filename=dp_obj.relative_filename,
            file_format=dp_obj.file_format,
            version=dp_obj.version,
            created=dp_obj.created,
            n_symbols=dp_obj.n_symbols
        )

        # Create a 1-row DataFrame with the new data
        new_row = pd.DataFrame([base_data | dp_obj.extra_metadata])

        df_meta = self.get_metadata()
        if not df_meta.size:
            df_meta = new_row
        else:
            df_meta = pd.concat([df_meta, new_row], axis=0)

        # Save to csv
        df_meta.to_csv(self.get_metadata_filename(), index=False)

    def save_data(self, data, datatype, frequency, file_format='parquet', **kwargs):
        """Save the data panel information."""
        dp_obj = PreprocessedDataPanel(db_helper=self, data=data, datatype=datatype, 
                                       frequency=frequency, file_format=file_format, **kwargs)
        self.save_data_from_object(dp_obj)

    def save_data_from_object(self, dp_obj):
        """Save the data panel information, given a PreprocessedDataPanel instance."""
        # Create the directory path if it does not exist
        path, _ = os.path.split(dp_obj.filename)
        if not os.path.isdir(path):
            os.makedirs(path)

        dp_obj.version = self._get_version_number(dp_obj.datatype)

        if os.path.isfile(dp_obj.filename):
            raise ValueError('Method should not be overwriting exciting file.')

        try:
            if dp_obj.file_format == 'parquet':
                fastparquet.write(dp_obj.filename, dp_obj.data)
            else:
                raise NotImplementedError(f"Unsupported file format: {dp_obj.file_format}")
            self.append_metadata(dp_obj)
        except Exception as e:  # pylint: disable=invalid-name,broad-exception-caught
            print('Failed to save file: ' + str(e))

            # Rollback file write in case it succeeded but metadata update failed
            if os.path.isfile(dp_obj.filename):
                os.remove(dp_obj.filename)

    def get_panel_data(self, datatype: str, version: int = None, **kwargs) -> pd.DataFrame:
        """Get a DataFrame containing the requested time series panel."""
        df_meta = self.get_metadata()
        if not df_meta.size:
            return pd.DataFrame([])

        for name, value in ({'datatype': datatype} | kwargs).items():
            mask = df_meta[name].values == value
            if not np.any(mask):
                return pd.DataFrame([])
            df_meta = df_meta.loc[mask]

        if version is not None:
            df_meta_ver = df_meta.query(f'version == {version}')
            if not df_meta_ver.size:
                return pd.DataFrame([])
            elif df_meta_ver.shape[0] != 1:
                raise ValueError('Multiple entries found for this datatype/version.')
            else:
                meta_row = pd.Series(df_meta_ver.iloc[0])
        else:
            meta_row = pd.Series(df_meta.sort_values('version').iloc[-1])

        path = self.get_datatype_path(meta_row.datatype)
        filename = os.path.join(path, meta_row.relative_filename)
        print(filename)
        return pd.read_parquet(filename, engine='pyarrow')

    def _get_version_number(self, datatype):
        """Get the version number for a dataset"""
        df_meta = self.get_metadata()
        if not df_meta.size:
            return 0

        df_meta = df_meta.query(f'datatype == "{datatype}"')
        if not df_meta.size:
            return 0
        else:
            return np.max(df_meta.version.values) + 1


class PreprocessedDataPanel(object):
    """Class to help save preprocessed data panels."""

    def __init__(self, db_helper, data, datatype, frequency, file_format='parquet', **kwargs):
        self.data = data.sort_index()
        self.datatype = datatype
        self.frequency = frequency
        self.db_helper = db_helper
        self.file_format = file_format
        self.extra_metadata = kwargs
        self._version = None

        self.start = pd.Timestamp(self.data.index.values[0])
        self.end = pd.Timestamp(self.data.index.values[-1])
        self.created = pd.Timestamp.now()
        self.n_symbols = data.shape[1]

    @property
    def relative_filename(self):
        """Gets the filename relative to the base_path."""
        return f'{self.datatype}_{self.version}.{self.file_format}'

    @property
    def filename(self):
        """Get the full filename for where to save the data panel."""
        path = self.db_helper.get_datatype_path(self.datatype)
        return os.path.join(path, self.relative_filename)

    @property
    def version(self):
        """Property that specifies the data version for a particular panel"""
        return self._version

    @version.setter
    def version(self, ver):
        """Setter method for the data version."""
        self._version = ver

    def save_data(self):
        """Method to save the time series panel."""
        self.db_helper.save_data(self)
