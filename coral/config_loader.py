
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any, Protocol
import yaml


class ConfigLoader:
    def __init__(self, config_path: str):
        self._config_path = Path(config_path)
        self._config_data: Dict[str, Any] = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def _load_config(self) -> Dict[str, Any]:
        """ loads the yaml file into a dictionary. is called by the config_data property."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    @property
    def config_data(self) -> Dict[str, Any]:
        """ laods the config into the _config_data property if it does not exsits

            returns the config_data."""
        if self._config_data is None:
            self._config_data = self._load_config()
        return self._config_data

    @property
    def data_path(self) -> Path:
        """ all configs should have a data_path where all data is stored, this property returns that path as a Path object"""
        return Path(self.config_data['data_path'])
