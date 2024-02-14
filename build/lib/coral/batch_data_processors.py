
import multiprocessing as mp
import numpy as np
import pandas as pd
import polars as pl
from itertools import product

import yaml

from abc import ABC, abstractmethod

from typing import List, Dict, Union, Tuple, Literal, Any, NewType, Protocol
from coral.experimental_metadata import ExperimentMetaData, MetaDataFactory
from coral.config_loader import ConfigLoader
from coral.data_preprocessor import DataPreprocessor, BehaviorDataPreprocessor, PhotometryDataPreprocessor
from coral.processing_strategies import ProcessingStrategy

from pathlib import Path


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
