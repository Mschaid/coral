import pandas as pd
import polars as pl
import seaborn as sns
import multiprocessing as mp
from coral.experimental_metadata import ExperimentMetaData, MetaDataFactory
from coral.config_loader import ConfigLoader
from coral.processing_strategies import ProcessingStrategy, BehaviorProcessingStrategy, PhotometryProcessingStrategy
from coral.aggregators import BehaviorAggregationStrategy, PhotometryAggregationStrategy, aggregate_data
from coral.batch_data_processors import BatchBehaviorDataPreprocessor, BatchPhotometryDataPreprocessor
import multiprocessing as mp
import logging


def create_logger(log_directory):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    log_file_directory = log_directory / 'main_analysis.log'
    logger = logging.getLogger('main_analysis_logger')

    if not log_directory.exists():
        log_directory.mkdir()
    logging.basicConfig(
        filename=log_file_directory,
        level=logging.INFO,
        format=log_format
    )

    return logger


# set up behavior analysis
def behavior_analysis(configs, meta_data_factory, behavior_strategy_params, cpus):
    logger = logging.getLogger('main_analysis_logger')
    logger.info("Running behavior analysis")

    behavior_strategy = BehaviorProcessingStrategy(**behavior_strategy_params)

    batch_behavior_preprocessor = BatchBehaviorDataPreprocessor(
        meta_data_factory, behavior_strategy)

    batch_behavior_preprocessor.process_data(num_processors=cpus)
    logger.info("Behavior analysis complete")

    aggregate_data(configs, BehaviorAggregationStrategy)

    logger.info("Data aggregation complete for behavior data")


def photometry_analysis(configs, meta_data_factory, photometry_strategy_params, cpus):
    logger = logging.getLogger('main_analysis_logger')
    logger.info("Running photometry analysis")

    photometry_processing_strategy = PhotometryProcessingStrategy(
        **photometry_strategy_params)

    batch_photometry_preprocessor = BatchPhotometryDataPreprocessor(
        meta_data_factory, photometry_processing_strategy)

    batch_photometry_preprocessor.process_data(num_processors=cpus)
    logger.info("Photometry analysis complete")

    aggregate_data(configs=configs,
                   aggregation_strategy=PhotometryAggregationStrategy)
    logger.info("Data aggregation complete for photometry data")


def main():
    # configs
    CPUS = mp.cpu_count()-2
    EXPERIMENTAL_CONFIGS = '/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Mike/LHA_dopamine/LH_NAC_Headfix_FP/Photometry/Pav_Training/claire_cohort/conf/config.yaml'
    CONFIGS = ConfigLoader(EXPERIMENTAL_CONFIGS)

    # set up logging
    LOG_DIRECTORY = CONFIGS.data_path / 'analysis_logs'
    if not LOG_DIRECTORY.exists():
        LOG_DIRECTORY.mkdir()

    logger = create_logger(log_directory=LOG_DIRECTORY)

    # set up metadata factory
    META_DATA_FACTORY = MetaDataFactory(CONFIGS)
    # set up strategy params

    BEHAVIOR_STRATEGY_PARAMS = {
        'config_key': 'behavioral_events',
        'time_window': (-10, 20),
        'events': (('cue', 'lick'),
                   ('cue', 'encoder'),
                   ('reward', 'lick'),
                   ('reward', 'encoder')
                   )
    }

    PHOTOMETRY_STRATEGY_PARAMS = {
        'signal_correction': 'z_score',
        'events_to_exclude': ['encoder', 'lick'],
        'save_bool': True,
        'return_df_bool': False
    }
    logger.info(f"Behavior strategy params: {BEHAVIOR_STRATEGY_PARAMS}")
    logger.info(f"Photometry strategy params: {PHOTOMETRY_STRATEGY_PARAMS}")

    behavior_analysis(configs=CONFIGS, meta_data_factory=META_DATA_FACTORY,
                      behavior_strategy_params=BEHAVIOR_STRATEGY_PARAMS, cpus=CPUS)
    photometry_analysis(configs=CONFIGS, meta_data_factory=META_DATA_FACTORY,
                        photometry_strategy_params=PHOTOMETRY_STRATEGY_PARAMS, cpus=CPUS)

    logger.info(
        "Batch analysis complete and saved at {CONFIGS.data_path/'aggregated_data'}")


if __name__ == '__main__':
    logger = logging.getLogger('main_analysis_logger')
    logger.info("Starting batch analysis")
    main()
    logging.info("Done")
