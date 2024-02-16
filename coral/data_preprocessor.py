import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import (Any, Dict, List, Literal, NewType, Optional, Protocol,
                    Tuple, Union)

import h5py
import numpy as np
import pandas as pd
import polars as pl
import yaml

from coral.config_loader import ConfigLoader
from coral.experimental_metadata import ExperimentMetaData, MetaDataFactory

Event = NewType('Event', str)
EventToAlign = NewType('EventToAlign', str)


class DataPreprocessor(Protocol):

    def __init__(self, metadata: ExperimentMetaData):
        pass


class BehaviorDataPreprocessor(DataPreprocessor):

    """
    This class is used to load and format data from the guppy experiment from the individual experiments.

    """

    def __init__(self, metadata: ExperimentMetaData):
        self.metadata = metadata
        self._processed_data_df = None

    @property
    def processed_data_df(self) -> pd.DataFrame:
        """ returns the processed data as a pandas dataframe"""
        return self._processed_data_df

    def _load_hdf5_file_to_numpy(self, file_path: Path, keyword='timestamps') -> np.ndarray:
        """ loads the hdf5 file into a numpy array"""
        time_stamps = h5py.File(file_path, "r").get(keyword)
        return np.array(time_stamps)

    def _create_dict_from_config(self, config_key) -> Dict[str, np.ndarray]:
        """ creates a dictionary of the behavior files from the config file. 
        The keys are the names of the events and the values are the numpy arrays of the timestamps.

        Returns
        -------
        Dict[str, np.ndarray]

        """

        config_dict = {f.stem: self._load_hdf5_file_to_numpy(
            f) for f in self.metadata.behavior_files}

        for stem, event in self.metadata.config_data[config_key].items():
            if stem in config_dict.keys():
                config_dict[event] = config_dict[stem]
                config_dict.pop(stem)
        return config_dict

    def _pad_array_with_nan(self, array: np.array, length: int) -> np.array:
        """ pads the array with nan values to the specified length """
        new_arr = np.full(length, np.nan)
        new_arr[:array.shape[0]] = array
        return new_arr

    def generate_df_from_config_dict(self, config_key) -> pd.DataFrame:
        """ generates a dataframe from the config dictionary. 
        The dataframe is padded with nan values to the length of the longest array."""

        config_dict = self._create_dict_from_config(config_key=config_key)

        config_dict_values = [v.shape[0] for v in config_dict.values()]
        if config_dict_values:
            max_length = max([v.shape[0] for v in config_dict.values()])
        else:
            max_length = 0
        padded_behavior_dict = {k: self._pad_array_with_nan(
            v, max_length) for k, v in config_dict.items()}

        return pd.DataFrame(padded_behavior_dict)

    def _align_events(self, df, events: Tuple[Event, EventToAlign]) -> Dict[str, pd.DataFrame]:
        """ aligns the events to the specified event.
        Returns 
        -------
        Dict[str, pd.DataFrame]
        """
        # unpacks the tuple
        event, event_to_align = events
        # gets the array of the events
        event_array = df[event][np.where(df[event] != 0)[0]].dropna()
        event_to_align_array = df[event_to_align].dropna().to_numpy()

        # creates a dictionary of the aligned events by iterating through the event array
        align_events_dict = {}
        for i in range(event_array.shape[0]):
            arr = np.array(event_to_align_array - event_array[i])
            align_events_dict[i] = arr
            event_dict = {
                k: pd.Series(v) for k, v in align_events_dict.items()
            }
            new_df = pd.DataFrame(event_dict)

        return {
            f"{event_to_align}_aligned_to_{event}": new_df
        }

    def _calculate_mean_event_frequency(self, data: Dict[str, pd.DataFrame], time_window: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """ calculate frequency of events around specifc EPOCH and returns a dictionary of the mean frequencies of the events.

        Returns  
        -------
        Dict[str, np.ndarray]
        """
        # time is in seconds
        mean_frequencies = {}
        for key, df in data.items():

            arr = df[(df > time_window[0]) & (
                df < time_window[1])].to_numpy()  #
            arr = arr[np.logical_not(np.isnan(arr))]
            freq = np.histogram(arr, bins=155)[0] / len(df.columns)
            freq_convert = freq * 5
            mean_frequencies[key] = freq_convert

        return mean_frequencies

    def batch_calculate_mean_event_frequency(self, data: Dict[str, pd.DataFrame], time_window: Tuple[int, int], *events: Tuple[Event, EventToAlign]):
        """ calculates the mean event frequency for a batch of events.
        * args are the events to be calculated.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        logger = logging.getLogger('main_analysis_logger')

        results = {}
        for event in events:
            try:
                aligned_events = self._align_events(data, event)
                results.update(self._calculate_mean_event_frequency(
                    aligned_events, time_window))
            except KeyError:
                logger.warning(
                    'event %s not found for %s ', event, self.metadata.experiment_id)

        return results

    def _format_meta_df(self) -> pd.DataFrame:
        """ formats the metadata dictionary into a pandas dataframe"""
        data = self.metadata.data
        df = (
            pd.DataFrame(data, index=[0])
            .assign(subject=lambda df_: df_.subject.astype('int64'),
                    user=lambda df_: df_.user.astype('category'),
                    date=lambda df_: df_.date.astype('datetime64[ns]'),
                    time_recorded=lambda df_: df_.time.astype('datetime64[ns]')
                    )
            .drop(columns=['time'])
        )
        return df

    def aggregate_processed_results(self, data: Dict[str, np.ndarray], return_df: bool = True, save=True) -> Union[None, pd.DataFrame]:
        """ aggregates the processed results into a pandas dataframe and saves it as a parquet file.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            dictionary of the processed results
        return_df : bool, optional
            returns the dataframe if true, by default True
        save : bool, optional
            saves the dataframe as a parquet file, by default True
            file is saved into the experiment directory.

        Returns
        -------
        Union[None, pd.DataFrame]
            returns the dataframe if return_df is True
        """

        meta_df = self._format_meta_df()
        data_df = pd.DataFrame(data)
        joined_df = (data_df
                     .join(meta_df, how='outer')
                     .ffill()
                     # assign time column
                     .assign(time=np.linspace(-10, 20, 155))
                     )
        # print(joined_df.shape)
        # joined_df = joined_df.set_index('time').

        if not self._processed_data_df:
            self._processed_data_df = joined_df
        if save:

            self.processed_data_df.to_parquet(
                self.metadata.main_path / f'{self.metadata.experiment_id}_processed_behavior_data.parquet')
        if return_df:
            return self.processed_data_df
        else:
            return


class PhotometryDataPreprocessor(DataPreprocessor):
    """
    Class for processing photometry data.

    Args:
        metadata (ExperimentMetaData): The metadata for the experiment.

    Attributes:
        metadata (ExperimentMetaData): The metadata for the experiment.
        _processed_data_df (None): The processed data dataframe.
        _structure_event_combos (None): The combinations of structure and event.
        _z_score_data_paths (List[Path]): The paths for z-score data.
        _dff_score_data_paths (List[Path]): The paths for dff score data.

    Methods:
        _create_structure_event_combos: Creates combinations of structure and event for a given signal correction key.
        _fetch_signal_correction_paths: Fetches the signal correction paths for a given signal correction key.
        z_score_data_paths: Getter method for z_score_data_paths property.
        dff_score_data_paths: Getter method for dff_score_data_paths property.
        """

    def __init__(self, metadata: ExperimentMetaData):
        self.metadata = metadata
        self._processed_data_df = None
        self._structure_event_combos = None
        self._z_score_data_paths: List[Path] = None
        self._dff_score_data_paths: List[Path] = None

    @property
    def processed_data_df(self):
        ''' returns the processed data as a polars dataframe'''
        return self._processed_data_df

    def _create_structure_event_combos(self, signal_correction_key: Literal['z_score', 'dff']):
        """
        Creates combinations of events, structures, and signal correction keys based on the given parameters.

        Parameters:
            signal_correction_key (Literal['z_score', 'dff']): The type of signal correction key to use.

        Returns:
            List[str]: A list of combinations of events, structures, and signal correction keys.
        """
        events = self.metadata.config_data['behavioral_events'].values()
        structures = self.metadata.config_data['structures'].values()
        combos = [f'{event}_{structure}_{signal_correction_key}' for event,
                  structure in product(events, structures)]
        return combos

    @property
    def structure_event_combos(self):
        """
        Property function to get structure event combos.
        """
        if not self._structure_event_combos:
            self._structure_event_combos = {
                k: self._create_structure_event_combos(k) for k in ['z_score', 'dff']}
        return self._structure_event_combos

    def _fetch_signal_correction_paths(self, signal_correction_key: Literal['z_score', 'dff']):
        """
        Fetches the signal correction paths for a given signal correction key.

        Parameters:
            signal_correction_key (Literal['z_score', 'dff']): The type of signal correction key to use.

        Returns:
            List[Path]: A list of paths for the signal correction files.
        """
        keys = self.structure_event_combos[signal_correction_key]
        files = [f for f in self.metadata.guppy_output_path.iterdir() if any(
            key in f.name for key in keys) and not f.name.startswith('peak')]

        try:
            assert len(files) > 0
        except Exception as exc:
            raise KeyError(
                f'No files found for keys {keys} in {self.metadata.guppy_output_path}') from exc

        return files

    @property
    def z_score_data_paths(self):
        """
        Getter method for z_score_data_paths property.
        """
        if not self._z_score_data_paths:
            self._z_score_data_paths = self._fetch_signal_correction_paths(
                'z_score')
        return self._z_score_data_paths

    @property
    def dff_score_data_paths(self):
        """
        Getter method for dff_score_data_paths property.
        """
        if not self._dff_score_data_paths:
            self._dff_score_data_paths = self._fetch_signal_correction_paths(
                'dff')
        return self._dff_score_data_paths

    def _categorize_event(self, event: str, events: List[str]) -> str:
        for e in events:
            if e in event:
                return e

    def _categorize_from_stem(self, path, category: Literal['behavioral_events', 'structures']):
        event = path.stem
        result = self._categorize_event(
            event, self.metadata.config_data[category].values())
        return {category: result}

    def _experiment_categories(self, path):

        structure_category = self._categorize_from_stem(path, 'structures')
        event_category = self._categorize_from_stem(path, 'behavioral_events')
        return {**structure_category, **event_category}

    def _roll_and_downsample(self, df, rolling_size, downsample_factor):
        return (
            df
            .dropna(axis=1)
            .rolling(rolling_size, center=True)
            .mean()
            .reset_index(drop=True)
            .dropna()[::downsample_factor]
        )
    #

    def _read_and_format_data_from_path(self, path) -> pl.DataFrame:
        def _reformat_subject(value):
            return (int(value.replace('_', '')))

        pl.enable_string_cache()
        constant_cols_to_drop = ['mean', 'err']
        raw_data = pd.read_hdf(path)
        smoothed_data = self._roll_and_downsample(
            df=raw_data, rolling_size=1000, downsample_factor=100)
        
        polars_frame = pl.from_pandas(smoothed_data)
        cols_to_rename = [col for col in polars_frame.columns if col not in [
            'timestamps', 'mean', 'err']]
        trials = [f'{i+1}' for i in range(len(cols_to_rename))]
        rename_dict = {col: trial for col,
                       trial in zip(cols_to_rename, trials)}
        category_data = self._experiment_categories(path)
        category_data.update(**self.metadata.data)

        data = (
            polars_frame
            .drop(constant_cols_to_drop)
            .rename(rename_dict)
            .with_columns([
                pl.Series(col, [val]*len(polars_frame)) for col, val in category_data.items()
            ])
            .melt(id_vars=['timestamps', 'behavioral_events', 'structures', 'subject', 'user', 'date', 'time'], variable_name='trial', value_name='z_score')
            .with_columns([
                pl.col('timestamps').cast(pl.Float32),
                pl.col('z_score').cast(pl.Float32),
                pl.col('date').str.strptime(
                    pl.Date, '%m/%d/%Y').cast(pl.Datetime),
                pl.col('trial').cast(pl.Int32),
                pl.col('behavioral_events').cast(pl.Categorical),
                pl.col('structures').cast(pl.Categorical),
                pl.col('subject').cast(pl.Categorical).apply(
                    lambda x: _reformat_subject(x)),
            ])
            .sort(['trial', 'timestamps', 'date', 'subject', 'structures', 'behavioral_events'])
        )
        return data

    def process_photometry_data(self, signal_correction: Literal['z_score', 'dff'] = 'z_score', events_to_exclude: List[str] = None, save=True, return_df=False) -> Optional[pl.DataFrame]:
        if not events_to_exclude:
            events_to_exclude = ['lick', 'encoder']
        if signal_correction == 'z_score':
            data_paths = self.z_score_data_paths
        if signal_correction == 'dff':
            data_paths = self.dff_score_data_paths

        paths_to_read = [p for p in data_paths if not any(
            p.stem.startswith(event) for event in events_to_exclude)]
        data = pl.concat([self._read_and_format_data_from_path(path)
                         for path in paths_to_read])

        if self._processed_data_df is None:
            self._processed_data_df = data
        if save:
            self.processed_data_df.write_parquet(
                self.metadata.main_path / f'{self.metadata.experiment_id}_processed_photometry_data.parquet')
        if return_df:
            return self.processed_data_df
        else:
            return
