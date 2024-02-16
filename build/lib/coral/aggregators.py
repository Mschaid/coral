
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl

from coral.config_loader import ConfigLoader


class AggregationStrategy(ABC):
    def __init__(self, configs: ConfigLoader):
        pass

    @abstractmethod
    def make_aggregate_dir(self):
        pass

    @abstractmethod
    def aggregate_processed_results_into_df(self):
        pass

    @abstractmethod
    def save_to_aggregate_dir(self):
        pass


class BehaviorAggregationStrategy(AggregationStrategy):
    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    def make_aggregate_dir(self, name: str = 'aggregated_data'):
        aggregate_dir = self.configs.data_path / name
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        return aggregate_dir

    def aggregate_processed_results_into_df(self, search_name: str = 'processed_behavior_data', file_type='parquet') -> pd.DataFrame:
        found_files = self.configs.data_path.rglob(
            f'*{search_name}.{file_type}')
        return pd.concat([pd.read_parquet(f) for f in found_files])

    def save_to_aggregate_dir(self, file_name_to_save: str = 'aggregated_behavior_data', data: pd.DataFrame = None):
        if not data:
            aggregate_data = self.aggregate_processed_results_into_df()
            aggregate_data.to_parquet(
                self.aggregate_dir / f'{file_name_to_save}.parquet')

        if data:
            data.to_parquet(self.aggregate_dir /
                            f'{file_name_to_save}.parquet')


class PhotometryAggregationStrategy(AggregationStrategy):
    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    def make_aggregate_dir(self, name: str = 'aggregated_data'):
        aggregate_dir = self.configs.data_path / name
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        return aggregate_dir

    def aggregate_processed_results_into_df(self, search_name: str = 'processed_photometry_data', file_type='parquet') -> pl.DataFrame:
        pl.enable_string_cache()
        found_files = self.configs.data_path.rglob(
            f'*{search_name}.{file_type}')

        return pl.concat([pl.read_parquet(f) for f in found_files])

    def save_to_aggregate_dir(self, file_name_to_save: str = 'aggregated_photometry_data', data: pl.DataFrame = None):
        if not data:
            aggregate_data = self.aggregate_processed_results_into_df()
            aggregate_data.write_parquet(
                self.aggregate_dir / f'{file_name_to_save}.parquet')

        if data:
            data.write_parquet(self.aggregate_dir /
                               f'{file_name_to_save}.parquet')


def aggregate_data(configs: ConfigLoader, aggregation_strategy: AggregationStrategy, data=None):
    aggregation_strategy = aggregation_strategy(configs)
    if data:
        aggregation_strategy.save_to_aggregate_dir(data=data)
    else:
        aggregation_strategy.save_to_aggregate_dir()
