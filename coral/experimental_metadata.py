
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import yaml

from coral.config_loader import ConfigLoader


class ExperimentMetaData:
    """ This class is used to extract metadata from the guppy experiment. It is used by the DataPreprocessor class to load and format data from the individual experiments."""

    def __init__(self, configs: ConfigLoader, main_path: str):
        self.configs = configs
        self._main_path = main_path  # this is the path from tdt
        self._stores_list: Dict[int, pd.DataFrame] = None
        self._data: Dict[str, Any] = None
        self._guppy_paths: Dict[str, Path] = None
        self._behavioral_events = None
        self._behavior_files = None
        self._stores_list_frame = None

    @property
    def config_path(self) -> Path:
        """ this is the path to the yaml file that contains the metadata for the experiment. This should be located in the parent directory of the main_path."""
        return self.configs.config_path

    @property
    def config_data(self) -> Dict[str, Any]:
        return self.configs.config_data

    @property
    def main_path(self) -> Path:
        """ This is the path to the main folder of the experiment, the same path that you select for guppy analysis. 

        eg. /Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Shudi/LHA_dopamine/analyzed_photometry/cohort_1/4756-231109-111759
        """
        if not isinstance(self._main_path, Path):
            self._main_path = Path(self._main_path)
        return self._main_path

    @property
    def experiment_id(self) -> str:
        """ returns the name of the directory eg 4756-231109-111759"""
        return self.main_path.name

    def _get_output_paths(self) -> List[Path]:
        """
        Retrieve a list of output paths from the main path of the experiment.

        Returns:
            List[Path]: A list of output paths from the main path of the experiment.
        """
        sub_dirs = [path for path in self.main_path.iterdir(
        ) if path.is_dir() and 'output' in path.name]
        return list(sub_dirs)

    def _get_meta_data(self) -> Dict[str, Any]:
        """
            Extract metadata from 'StoresListing.txt' files in the main directory.

            This method iterates over all files in the main directory. If a file is named 'StoresListing.txt',
            it reads the file and extracts metadata from lines 2 to 5. Each line is expected to contain a key-value pair
            separated by ': '. The key-value pairs are stored in a dictionary and returned.

            Returns
            -------
            Dict[str, Any]
                A dictionary where each key is a string representing the metadata field name, and each value is the corresponding metadata value.
        """
        data = {}
        for file in self.main_path.iterdir():
            if file.is_file() and 'StoresListing.txt' in file.name:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    meta_data_lines = lines[1:5]
                    key_value_pairs = map(lambda line: line.strip(
                        '\n').split(': '), meta_data_lines)
                    data = {k.lower(): v for k, v in key_value_pairs}

        return data

    @property
    def data(self) -> Dict[str, Any]:
        """ returns the data property if it exsits, otherwise it calls the _get_meta_data method to extract the data from the main_path directory."""
        if self._data is None:
            self._data = self._get_meta_data()
        return self._data

    @property
    def behavioral_events(self) -> List[str]:
        """ returns a list of behavioral events from the config_data property."""
        if not self._behavioral_events:
            self._behavioral_events = list(
                self.configs.config_data['behavioral_events'].keys())
        return self._behavioral_events

    # @property
    # def photmetry

    def _get_stores_list_paths(self) -> List[Path]:
        """ returns a list of storesList.csv files from the output_paths property."""
        output_paths = self._get_output_paths()
        stores_lists_paths = [output_path /
                              'storesList.csv' for output_path in output_paths]
        return stores_lists_paths

    def _get_paths_by_filetype(self, file_type: str = 'hdf5') -> List[Path]:
        """ returns a list of paths for a given file type."""
        files = [f for f in self.main_path.glob(f'**/*.{file_type}')]
        return files

    def _filter_files_by_keywords(self, files: List[Path], keywords: List[str]) -> List[Path]:
        """ returns a list of files that contain the keywords in their name."""

        filtered_files = [f for f in files if any(
            k in f.name for k in keywords)]
        return filtered_files

    @property
    def behavior_files(self):
        """ returns a list of behavior files from the main_path directory."""
        if not self._behavior_files:
            hdf5_files = self._get_paths_by_filetype(file_type='hdf5')
            self._behavior_files = self._filter_files_by_keywords(
                hdf5_files, self.behavioral_events)
        return self._behavior_files

    @property
    def guppy_paths(self) -> Dict[str, Path]:
        """ returns a dictionary of paths for the experiment."""
        if self._guppy_paths is None:
            self._guppy_paths = {}
            self._guppy_paths['output_paths'] = self._get_output_paths()
            self._guppy_paths['stores_lists_paths'] = self._get_stores_list_paths()
            self._guppy_paths['behavior_files'] = self.behavior_files
            # self._guppy_paths[]
        return self._guppy_paths

    @property
    def guppy_output_path(self) -> Path:
        """ returns the output path for the guppy experiment. If there are multiple output paths, it returns the first one."""
        if len(self.guppy_paths['output_paths']) < 2:
            return self.guppy_paths['output_paths'][0]
        return self.guppy_paths['output_paths']


class MetaDataFactory:
    """ This class is used to create ExperimentMetaData objects for each experiment in the data_path directory. It is used by the BatchPhotometryDataPreprocessor class to create a list of ExperimentMetaData objects."""
    def __init__(self, configs: ConfigLoader):
        self.configs = configs

    @property
    def data_path(self):
        return Path(self.configs.config_data['data_path'])

    def fetch_batch_metadata(self):
        return [d for d in self.data_path.iterdir() if d.is_dir()]

    @property
    def all_meta_data(self):
        paths_to_igore = ["average", "aggregated_data", "analysis_logs"]
        return [ExperimentMetaData(self.configs, d) for d in self.fetch_batch_metadata() if d.name not in paths_to_igore]
