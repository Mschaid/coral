from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl

from coral.config_loader import ConfigLoader


class AggregationStrategy(ABC):
    """An abstract base class representing an aggregation strategy.

    Attributes:
        configs (ConfigLoader): An instance of ConfigLoader containing configuration settings.

    """

    def __init__(self, configs: ConfigLoader):
        pass

    @abstractmethod
    def make_aggregate_dir(self):
        """Abstract method to create a directory for aggregated data."""
        pass

    @abstractmethod
    def aggregate_processed_results_into_df(self):
        """Abstract method to aggregate processed results into a DataFrame."""
        pass

    @abstractmethod
    def save_to_aggregate_dir(self):
        """Abstract method to save aggregated data to the aggregate directory."""
        pass


class BehaviorAggregationStrategy(AggregationStrategy):
    """A concrete implementation of AggregationStrategy for behavior data.

    Attributes:
        configs (ConfigLoader): An instance of ConfigLoader containing configuration settings.
        aggregate_dir (Path): Path to the directory for aggregated data.
    """

    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    def make_aggregate_dir(self, name: str = 'aggregated_data'):
        """Create a directory for aggregated data.

        Args:
            name (str, optional): Name of the directory. Defaults to 'aggregated_data'.

        Returns:
            Path: Path to the created directory.
        """
        aggregate_dir = self.configs.data_path / name
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        return aggregate_dir

    def aggregate_processed_results_into_df(self, search_name: str = 'processed_behavior_data', file_type='parquet') -> pd.DataFrame:
        """Aggregate processed behavior data into a pandas DataFrame.

        Args:
            search_name (str, optional): Name pattern for processed behavior data files. Defaults to 'processed_behavior_data'.
            file_type (str, optional): File type of processed data files. Defaults to 'parquet'.

        Returns:
            pd.DataFrame: Aggregated DataFrame of processed behavior data.
        """
        found_files = self.configs.data_path.rglob(
            f'*{search_name}.{file_type}')
        return pd.concat([pd.read_parquet(f) for f in found_files])

    def save_to_aggregate_dir(self, file_name_to_save: str = 'aggregated_behavior_data', data: pd.DataFrame = None):
        """Save aggregated behavior data to the aggregate directory.

        Args:
            file_name_to_save (str, optional): Name of the file to save. Defaults to 'aggregated_behavior_data'.
            data (pd.DataFrame, optional): DataFrame to save. Defaults to None.
        """
        if not data:
            aggregate_data = self.aggregate_processed_results_into_df()
            aggregate_data.to_parquet(
                self.aggregate_dir / f'{file_name_to_save}.parquet')

        if data:
            data.to_parquet(self.aggregate_dir /
                            f'{file_name_to_save}.parquet')


class PhotometryAggregationStrategy(AggregationStrategy):
    """A concrete implementation of AggregationStrategy for photometry data.

    Attributes:
        configs (ConfigLoader): An instance of ConfigLoader containing configuration settings.
        aggregate_dir (Path): Path to the directory for aggregated data.
    """

    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    def make_aggregate_dir(self, name: str = 'aggregated_data'):
        """Create a directory for aggregated data.

        Args:
            name (str, optional): Name of the directory. Defaults to 'aggregated_data'.

        Returns:
            Path: Path to the created directory.
        """
        aggregate_dir = self.configs.data_path / name
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        return aggregate_dir

    def aggregate_processed_results_into_df(self, search_name: str = 'processed_photometry_data', file_type='parquet') -> pl.DataFrame:
        """Aggregate processed photometry data into a Polars DataFrame.

        Args:
            search_name (str, optional): Name pattern for processed photometry data files. Defaults to 'processed_photometry_data'.
            file_type (str, optional): File type of processed data files. Defaults to 'parquet'.

        Returns:
            pl.DataFrame: Aggregated DataFrame of processed photometry data.
        """
        pl.enable_string_cache()
        found_files = self.configs.data_path.rglob(
            f'*{search_name}.{file_type}')
        return pl.concat([pl.read_parquet(f) for f in found_files])

    def save_to_aggregate_dir(self, file_name_to_save: str = 'aggregated_photometry_data', data: pl.DataFrame = None):
        """Save aggregated photometry data to the aggregate directory.

        Args:
            file_name_to_save (str, optional): Name of the file to save. Defaults to 'aggregated_photometry_data'.
            data (pl.DataFrame, optional): DataFrame to save. Defaults to None.
        """
        if not data:
            aggregate_data = self.aggregate_processed_results_into_df()
            aggregate_data.write_parquet(
                self.aggregate_dir / f'{file_name_to_save}.parquet')

        if data:
            data.write_parquet(self.aggregate_dir /
                               f'{file_name_to_save}.parquet')


def aggregate_data(configs: ConfigLoader, aggregation_strategy: AggregationStrategy, data=None):
    """Aggregate data using a specified aggregation strategy.

    Args:
        configs (ConfigLoader): An instance of ConfigLoader containing configuration settings.
        aggregation_strategy (AggregationStrategy): Aggregation strategy to use.
        data (optional): Data to be aggregated. Defaults to None.
    """
    aggregation_strategy = aggregation_strategy(configs)
    if data:
        aggregation_strategy.save_to_aggregate_dir(data=data)
    else:
        aggregation_strategy.save_to_aggregate_dir()
# a random comment 
