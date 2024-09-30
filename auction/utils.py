from pathlib import Path
import logging
from logging import Logger
import json

import pandas as pd


def _default_formatter() -> logging.Formatter:
    """
    Default formatter
    :return: formatter
    """
    return logging.Formatter('[{asctime}][{levelname}] {message}', '%Y-%m-%d %H:%M:%S', style='{')


def _default_logger(name: str) -> Logger:
    """
    Default logger
    :param name: name of logger
    :return: logger
    """
    logger = Logger(name)
    logger.setLevel(logging.INFO)
    return logger


def console_logger(name: str) -> Logger:
    """
    Default console logger
    :param name: name of logger
    :return: logger
    """
    logger = _default_logger(name)
    formatter = _default_formatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_train_report(exp_name: str) -> pd.DataFrame:
    """
    Parse catboost logs
    :param exp_name: experiment name and tag
    :return: pd.DataFrame with metrics
    """
    p = Path(f'./model_logs/{exp_name}/catboost_training.json')
    with p.open() as file_handler:
        metrics = json.load(file_handler)
    df = pd.DataFrame(metrics['iterations'])
    metrics_names = [x['name'] for x in metrics['meta']['learn_metrics']]
    test_metrics_names = [f'Test_{x}' for x in metrics_names]
    train_metrics_names = [f'Train_{x}' for x in metrics_names]

    df.loc[:, test_metrics_names] = df.loc[:, 'test'].to_list()
    df.loc[:, train_metrics_names] = df.loc[:, 'learn'].to_list()

    out_cols = sum([list(x) for x in zip(train_metrics_names, test_metrics_names)], [])

    return df.loc[:, ['iteration'] + out_cols]
