
import multiprocessing as mp
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, NewType, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import yaml

from coral.config_loader import ConfigLoader
from coral.data_preprocessor import (BehaviorDataPreprocessor,
                                     DataPreprocessor,
                                     PhotometryDataPreprocessor)
from coral.experimental_metadata import ExperimentMetaData, MetaDataFactory
from coral.processing_strategies import ProcessingStrategy


class BatchDataPreprocessor(ABC):
    @abstractmethod
    def __init__(self, metadata_factory: MetaDataFactory, processing_strategy: ProcessingStrategy):
        pass

    @abstractmethod
    def preprocessor_factory(self):
        pass

    @abstractmethod
    def process_data(self, num_processors: int = 2) -> None:
        pass


class BatchBehaviorDataPreprocessor(BatchDataPreprocessor):

    def __init__(self, metadata_factory: MetaDataFactory, processing_strategy: ProcessingStrategy):
        self.metadata_factory = metadata_factory
        self.processing_strategy = processing_strategy

    def preprocessor_factory(self) -> List[BehaviorDataPreprocessor]:
        """ creates a data preprocessor for a each experiment"""
        return [BehaviorDataPreprocessor(metadata) for metadata in self.metadata_factory.all_meta_data]

    def process_data(self, num_processors: int = 2) -> None:
        """ processes the data for each experiment in parallel

        Attributes
        ----------
        num_processors : int
            number of processors to use, by default 2
        """

        pool = mp.Pool(processes=num_processors)
        preprocessors = self.preprocessor_factory()
        pool.map(self.processing_strategy.process, preprocessors)
        pool.close()
        pool.join()


class BatchPhotometryDataPreprocessor(BatchDataPreprocessor):
    def __init__(self, metadata_factory: MetaDataFactory, processing_strategy: ProcessingStrategy):
        self.metadata_factory = metadata_factory
        self.processing_strategy = processing_strategy

    def preprocessor_factory(self) -> List[PhotometryDataPreprocessor]:
        '''creates a photometry data preprocessor for each experiment'''
        return [PhotometryDataPreprocessor(metadata) for metadata in self.metadata_factory.all_meta_data]

    def process_data(self, num_processors: int = 2) -> None:

        pool = mp.Pool(processes=num_processors)
        preprocessors = self.preprocessor_factory()
        pool.map(self.processing_strategy.process, preprocessors)
        pool.close()
        pool.join()
