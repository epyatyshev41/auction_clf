from abc import abstractmethod
from logging import Logger
from typing import Dict

import pandas as pd
import numpy as np

from auction.base import StrategyAuction
from auction.constants import UNIQ_COL
from auction.utils import console_logger


class BaseSplitter(StrategyAuction):
    """Base class for splitter building"""

    def __init__(self, params: Dict, logger: Logger = console_logger('')):
        """
        Init method
        :param params: parameters dict for data splitter
        :param logger:
        """
        super().__init__(params=params, logger=logger)

    @abstractmethod
    def split(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Return uniq ids for train and test
        """
        raise NotImplementedError


class RandomSplitter(BaseSplitter):
    """
    Split randomly
    Example, which splits 70% to train, 30% to test:
    "splitter": {
        "strategy": "random",
        "params": {
          "train_test_ratio": 0.7,
          "seed": 5,
          "uniq_col": "ifa"
        }
    """
    strategy = 'random'

    def __init__(self, params: Dict, logger: Logger = console_logger('')):
        super().__init__(params=params, logger=logger)
        self.train_test_ratio = params['train_test_ratio']
        self.seed = params['seed']
        self.uniq_col = params.get('uniq_col', UNIQ_COL)

    def split(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Select `self.train_test_ratio` random part of all inputs ids
        :param data: input data
        :return: uniq ids for train and test
        """
        ids = data.loc[:, self.uniq_col].values.astype(str)
        uniq_ids = np.unique(ids)
        train_size = int(self.train_test_ratio * len(uniq_ids))
        np.random.seed(self.seed)
        train_ids = np.random.choice(uniq_ids, size=train_size, replace=False)
        left_ids = np.setdiff1d(uniq_ids, train_ids)
        test_ids = left_ids
        return train_ids, test_ids