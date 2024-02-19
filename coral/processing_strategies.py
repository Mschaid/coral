from abc import ABC, abstractmethod
from typing import List, Literal, NewType, Tuple

from coral.data_preprocessor import (BehaviorDataPreprocessor,
                                     DataPreprocessor,
                                     PhotometryDataPreprocessor)

Event = NewType('Event', str)
EventToAlign = NewType('EventToAlign', str)


class ProcessingStrategy(ABC):
    """ An abstract class for processing strategies. """

    @abstractmethod
    def process(self, data_preprocessor: DataPreprocessor):
        pass


class BehaviorProcessingStrategy(ProcessingStrategy):
    """
    concrete implementation of the ProcessingStragegy abstract class for processing behavior data.
    
    Attributes:
        config_key (str): the config key to use for the config file
        time_window (Tuple[int, int]): the time window to use for the mean event frequency calculation
        events (Tuple[Tuple[Event, EventToAlign]]): the events to use for the mean event frequency calculation
        return_df (bool): whether to return the dataframe or not
    """
    def __init__(self, config_key: str, time_window: Tuple[int, int], events: Tuple[Tuple[Event, EventToAlign]], return_df: bool = False):
        self.config_key = config_key
        self.time_window = time_window
        self.events = events
        self.return_df = return_df

    def process(self, data_preprocessor: BehaviorDataPreprocessor, return_dfs: bool = False):
        """
        method to process the behavior data. This method will call the generate_df_from_config_dict method from the DataPreprocessor class to generate a dataframe from the config file. It will then call the batch_calculate_mean_event_frequency method to calculate the mean event frequency for the events in the time window. Finally, it will call the aggregate_processed_results method to aggregate the results.
        """

        behavior_df = data_preprocessor.generate_df_from_config_dict(
            config_key='behavioral_events')

        mean_dict = data_preprocessor.batch_calculate_mean_event_frequency(
            behavior_df, self.time_window, *self.events)

        data_preprocessor.aggregate_processed_results(
            mean_dict, return_df=self.return_df)


class PhotometryProcessingStrategy(ProcessingStrategy):
    """
    concrete implementation of the ProcessingStragegy abstract class for processing photometry data.
    
    Attributes:
        signal_correction (Literal['z_score', 'dff']): the signal correction to use
        events_to_exclude (List[Event]): the events to exclude
        save (bool): whether to save the dataframe
        return_df (bool): whether to return the dataframe or not
    """
    def __init__(self, signal_correction: Literal['z_score', 'dff'], events_to_exclude: List[Event], save_bool: bool, return_df_bool: bool):
        self.signal_correction = signal_correction
        self.events_to_exclude = events_to_exclude
        self.save_bool = save_bool
        self.return_df_bool = return_df_bool

    def process(self, data_preprocessor: PhotometryDataPreprocessor):
        """
        method to process the photometry data. This method will call the process_photometry_data method from the DataPreprocessor class to process the photometry data.
        """
        data_preprocessor.process_photometry_data(signal_correction=self.signal_correction,
                                                  events_to_exclude=self.events_to_exclude,
                                                  save=self.save_bool,
                                                  return_df=self.return_df_bool)
